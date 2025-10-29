package com.auto_fe.auto_fe.ui.screens.layers

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import com.auto_fe.auto_fe.ui.components.OrbAISphere
import com.auto_fe.auto_fe.ui.theme.*

/**
 * BackgroundLayer - Orb Sphere + Dynamic Gradient Background
 * 
 * Refactored Features:
 * - Smooth gradient background cho depth
 * - Ultra-smooth Orb AI Sphere
 * - Optimized performance với layered rendering
 * - Minimalist & impressive visual design
 */
@Composable
fun BackgroundLayer(
    voiceLevel: Int,
    isListening: Boolean = false,
    isDarkMode: Boolean = true,
    performancePreset: String = "balanced",
    onSphereClick: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(
                brush = Brush.verticalGradient(
                    colors = if (isDarkMode) {
                        listOf(
                            AIBackgroundDeep,
                            AIBackgroundMid,
                            AIBackgroundDeep
                        )
                    } else {
                        listOf(
                            Color(0xFFFAFAFA), // Very light white
                            Color(0xFFFFFFFF), // Pure white
                            Color(0xFFFAFAFA)  // Very light white
                        )
                    }
                )
            )
    ) {
        // Ultra-smooth AI Visual Sphere
        OrbAISphere(
            voiceLevel = voiceLevel,
            isListening = isListening,
            isDarkMode = isDarkMode,
            performancePreset = performancePreset,
            onClick = onSphereClick,
            modifier = Modifier.fillMaxSize()
        )
    }
}