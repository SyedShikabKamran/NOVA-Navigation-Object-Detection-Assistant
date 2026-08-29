package com.nova.assistant.data.local

import androidx.room.*
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.flow.Flow

// ─────────────────────────────────────────────
// ENTITIES
// ─────────────────────────────────────────────

@Entity(tableName = "room_snapshots")
data class RoomSnapshotEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val roomName: String,
    val objectInventoryJson: String,   // JSON: [{class, direction, count}]
    val createdAt: Long,
    val wifiFingerprint: String = "",  // JSON WiFi fingerprint at scan time
    val scanHeadingDeg: Float = -1f    // compass bearing at scan time; -1=unknown
)

@Entity(tableName = "settings")
data class SettingsEntity(
    @PrimaryKey val key: String,
    val value: String
)

// ─────────────────────────────────────────────
// DAOs
// ─────────────────────────────────────────────

@Dao
interface RoomSnapshotDao {
    @Insert
    suspend fun insert(room: RoomSnapshotEntity): Long

    @Query("SELECT * FROM room_snapshots ORDER BY createdAt DESC")
    fun getAll(): Flow<List<RoomSnapshotEntity>>

    @Query("SELECT * FROM room_snapshots WHERE roomName = :name LIMIT 1")
    suspend fun findByName(name: String): RoomSnapshotEntity?

    @Query("DELETE FROM room_snapshots WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("UPDATE room_snapshots SET roomName = :name WHERE id = :id")
    suspend fun renameById(id: Long, name: String)

    @Query("DELETE FROM room_snapshots")
    suspend fun deleteAll()
}

@Dao
interface SettingsDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun set(setting: SettingsEntity)

    @Query("SELECT value FROM settings WHERE `key` = :key LIMIT 1")
    suspend fun get(key: String): String?

    @Query("SELECT * FROM settings")
    fun getAll(): Flow<List<SettingsEntity>>

    @Query("DELETE FROM settings WHERE `key` = :key")
    suspend fun delete(key: String)
}

// ─────────────────────────────────────────────
// DATABASE
// ─────────────────────────────────────────────

@Database(
    entities = [
        RoomSnapshotEntity::class,
        SettingsEntity::class,
    ],
    version = 6,
    exportSchema = false
)
abstract class NovaDatabase : RoomDatabase() {
    abstract fun roomSnapshotDao(): RoomSnapshotDao
    abstract fun settingsDao(): SettingsDao

    companion object {
        const val DATABASE_NAME = "nova_db"

        // ponytail: MIGRATION_1_2 intentionally omitted — DB version 1 was never publicly
        //   distributed. fallbackToDestructiveMigration() covers any stale dev installs.
        //   If this app ships to real users, add MIGRATION_1_2 before removing the fallback.

        // Additive migration: new columns all have DB-level defaults so existing rows are valid.
        // Note: expiresAt DEFAULT 0 is intentional — HazardDao.findNearby() uses isManual to
        //   distinguish "no expiry" (manual) from "pre-migration auto" (isManual=0, expiresAt=0).
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE hazards ADD COLUMN expiresAt INTEGER NOT NULL DEFAULT 0")
                database.execSQL("ALTER TABLE hazards ADD COLUMN wifiFingerprint TEXT NOT NULL DEFAULT ''")
                database.execSQL("ALTER TABLE room_snapshots ADD COLUMN wifiFingerprint TEXT NOT NULL DEFAULT ''")
                database.execSQL("ALTER TABLE room_snapshots ADD COLUMN scanHeadingDeg REAL NOT NULL DEFAULT -1")
            }
        }

        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE hazards ADD COLUMN label TEXT NOT NULL DEFAULT ''")
            }
        }

        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("DROP TABLE IF EXISTS training_samples")
            }
        }

        // Hazard Memory feature removed entirely (2026-07-14) — drops the table rather than
        // fallbackToDestructiveMigration()'s whole-DB wipe, so existing room_snapshots survive.
        val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("DROP TABLE IF EXISTS hazards")
            }
        }
    }
}

// ─────────────────────────────────────────────
// SETTING KEYS
// ─────────────────────────────────────────────

object SettingsKeys {
    const val VISION_MODE = "vision_mode"           // "full_blind" | "partial"
    const val SPEECH_RATE = "speech_rate"           // "slow" | "normal" | "fast"
    const val VIBRATION = "vibration"               // "off" | "gentle" | "strong"
    const val WARNING_DISTANCE = "warning_distance" // "1" | "2" | "3"
    const val EMERGENCY_CONTACT = "emergency_contact"
    const val EMERGENCY_NAME = "emergency_name"
    const val USER_NAME = "user_name"
    const val SETUP_COMPLETE = "setup_complete"
    // C6/R6: Version-scoped key — changing "v1" here resets the flag for all users
    // on the next version bump. Prevents silent skip of startup TTS after a backup
    // restore (Android auto-backup would restore an old "startup_announced" value,
    // but a new key like "startup_announced_v1" would not exist in the backup and
    // therefore would not be restored, ensuring the announcement fires on fresh installs).
    const val STARTUP_ANNOUNCED = "startup_announced_v1"
    // R10: Alive signal interval — "2" | "5" | "10" | "off". Defaults to "5" (5 minutes)
    // if the key is absent. Stored as a string to match the pattern used by other settings.
    const val ALIVE_SIGNAL_INTERVAL = "alive_signal_interval"
    // Shown once on first launch when device heap < 1.5 GB (proxy for ≤ 2 GB RAM).
    const val DEVICE_PERF_WARNED = "device_perf_warned"
}
