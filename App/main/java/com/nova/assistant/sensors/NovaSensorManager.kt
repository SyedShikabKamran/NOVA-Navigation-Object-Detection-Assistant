package com.nova.assistant.sensors

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.BatteryManager
import com.nova.assistant.engine.SensorData
import com.nova.assistant.util.NovaConstants
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.math.sqrt

/**
 * Aggregates all phone sensor data into a single SensorData stream.
 *
 * Tracks:
 * - Motion (accelerometer) — for adaptive frame rate
 * - Light (lux) — for low-light detection
 * - Tilt (gyroscope) — for camera angle compensation
 * - Battery — for power management
 * - Thermal — for thermal throttling prevention
 */
@Singleton
class NovaSensorManager @Inject constructor(
    private val context: Context
) : SensorEventListener {

    private val sensorManager: SensorManager =
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val batteryManager: BatteryManager =
        context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    private val gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
    private val rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val _sensorData = MutableStateFlow(SensorData())
    val sensorData: StateFlow<SensorData> = _sensorData

    // Internal state
    private var lastMotionMagnitude = 0f
    private var lastLightLux = 200f
    private var lastTiltDegrees = 0f
    private var lastAzimuthDeg = -1f
    private var motionHistory = mutableListOf<Float>()

    // Battery temperature cached to avoid calling registerReceiver on every sensor event.
    // Sensor events fire at ~20Hz; temperature changes <1Hz, so a 10s cache is safe.
    @Volatile private var lastTemperatureCelsius = 25f
    private var lastTempUpdateMs = 0L

    // Idempotency guard — prevents double-registration if start() is called twice
    // (e.g. ViewModel re-create or service restart race condition).
    private var isStarted = false

    fun start() {
        if (isStarted) return
        isStarted = true
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        lightSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        gravity?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        rotationVector?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    fun stop() {
        if (!isStarted) return
        isStarted = false
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                val (x, y, z) = event.values
                val magnitude = sqrt(x * x + y * y + z * z) - SensorManager.GRAVITY_EARTH
                lastMotionMagnitude = kotlin.math.abs(magnitude)

                // Keep rolling average over last ~30 readings
                motionHistory.add(lastMotionMagnitude)
                if (motionHistory.size > 30) motionHistory.removeAt(0)
            }
            Sensor.TYPE_LIGHT -> {
                lastLightLux = event.values[0]
            }
            Sensor.TYPE_GRAVITY -> {
                // Use gravity to determine phone tilt (more accurate than gyroscope alone)
                val (gx, gy, gz) = event.values
                val tilt = Math.toDegrees(
                    Math.atan2(gx.toDouble(), sqrt((gy * gy + gz * gz).toDouble()))
                ).toFloat()
                lastTiltDegrees = tilt
            }
            Sensor.TYPE_GYROSCOPE -> {
                // Used for fine motion detection if needed
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val rotMatrix = FloatArray(9)
                val orient = FloatArray(3)
                SensorManager.getRotationMatrixFromVector(rotMatrix, event.values)
                SensorManager.getOrientation(rotMatrix, orient)
                // orient[0] = azimuth in radians; convert to 0..360°
                var az = Math.toDegrees(orient[0].toDouble()).toFloat()
                if (az < 0f) az += 360f
                lastAzimuthDeg = az
            }
        }

        updateSensorData()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * Compose all sensor data into a single state.
     */
    private fun updateSensorData() {
        val avgMotion = if (motionHistory.isNotEmpty()) motionHistory.average().toFloat() else 0f
        val isMoving = avgMotion > NovaConstants.MOTION_THRESHOLD

        val batteryLevel = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)

        // Rate-limit registerReceiver to at most once per 10 seconds.
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastTempUpdateMs > 10_000L) {
            lastTemperatureCelsius = getBatteryTemperature()
            lastTempUpdateMs = nowMs
        }

        _sensorData.value = SensorData(
            isMoving = isMoving,
            lightLux = lastLightLux,
            tiltAngleDegrees = lastTiltDegrees,
            batteryPercent = batteryLevel,
            temperatureCelsius = lastTemperatureCelsius,
            motionMagnitude = lastMotionMagnitude,
            azimuthDeg = lastAzimuthDeg
        )
    }

    /**
     * Battery temperature (proxy for phone temperature).
     */
    private fun getBatteryTemperature(): Float {
        val batteryStatus = context.registerReceiver(
            null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )
        // Temperature is in tenths of a degree Celsius
        val tempTenths = batteryStatus?.getIntExtra(
            BatteryManager.EXTRA_TEMPERATURE, 250
        ) ?: 250
        return tempTenths / 10f
    }

    fun getCurrentSensorData(): SensorData = _sensorData.value
}
