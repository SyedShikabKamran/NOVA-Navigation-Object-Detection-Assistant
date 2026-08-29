package com.nova.assistant.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val NovaDarkColors = darkColorScheme(
    primary = Color(0xFF00E5FF),         // Cyan — high contrast
    onPrimary = Color.Black,
    secondary = Color(0xFFFFD600),       // Yellow — visible warnings
    onSecondary = Color.Black,
    tertiary = Color(0xFFFF5252),        // Red — danger
    background = Color.Black,            // OLED black for battery
    onBackground = Color.White,
    surface = Color(0xFF121212),
    onSurface = Color.White,
    error = Color(0xFFFF1744),
    onError = Color.White
)

private val NovaLightColors = lightColorScheme(
    primary = Color(0xFF006B8E),
    onPrimary = Color.White,
    secondary = Color(0xFF8A6F00),
    background = Color.White,
    onBackground = Color.Black
)

@Composable
fun NovaTheme(content: @Composable () -> Unit) {
    // Always dark for VIP: OLED battery savings + low-vision-friendly
    MaterialTheme(
        colorScheme = NovaDarkColors,
        content = content
    )
}
