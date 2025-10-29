package com.auto_fe.auto_fe.ui.theme

import androidx.compose.ui.graphics.Color

// ===== AI VISUAL IDENTITY - MINIMALIST, DEEP-SOFT =====
// Tone chủ đạo: Minimalist, Deep-Soft, cảm giác "AI sống động"
// Mood board: Glow layer, Glass aura, Orbit motion, Gradient soft blue–purple, Pulsating sphere

// ===== PRIMARY AI COLORS (Soft Blue-Purple Gradient) =====
val AIPrimarySoft       = Color(0xFF6B73FF)   // Soft blue-purple
val AIPrimaryGlow       = Color(0xFF8B9AFF)   // Glow layer
val AIPrimaryGlass      = Color(0xFF9BB5FF)   // Glass aura
val AIPrimaryDeep       = Color(0xFF4A5BCC)   // Deep soft

// ===== BACKGROUND SYSTEM (Deep-Soft Minimalist) =====
val AIBackgroundDeep    = Color(0xFF0A0B0F)   // Deep black-blue
val AIBackgroundMid     = Color(0xFF1A1D2B)   // Mid dark (for gradients)
val AIBackgroundSoft    = Color(0xFF1A1D2B)   // Soft dark
val AISurfaceGlass      = Color(0xFF2A2F42)   // Glass surface
val AISurfaceGlow       = Color(0xFF3A4155)   // Glow surface

// ===== LIGHT MODE BACKGROUND SYSTEM =====
val LightBackgroundDeep = Color(0xFFF8F6F0)   // Warm white-beige
val LightBackgroundMid  = Color(0xFFF5F2ED)   // Mid beige (for gradients)
val LightBackgroundSoft = Color(0xFFF2EFEA)   // Soft beige
val LightSurfaceGlass   = Color(0xFFF0EDE8)   // Glass surface
val LightSurfaceGlow    = Color(0xFFEDEAE5)   // Glow surface

// ===== TEXT SYSTEM (Minimalist, ít chữ) =====
val AITextPrimary       = Color(0xFFE8F0FF)   // Soft white-blue
val AITextSecondary     = Color(0xFFB8C5E8)   // Muted soft
val AITextTertiary      = Color(0xFF8A9BC2)   // Very muted

// ===== LIGHT MODE TEXT SYSTEM =====
val LightTextPrimary    = Color(0xFF2C2C2C)   // Dark grey
val LightTextSecondary  = Color(0xFF5A5A5A)   // Medium grey
val LightTextTertiary   = Color(0xFF8A8A8A)   // Light grey

// ===== VOICE RESPONSIVE COLORS (Breathing with voice) =====
val VoiceIdle           = Color(0xFF4A5BCC)   // Deep soft blue
val VoiceListening      = Color(0xFF6B73FF)   // Soft blue-purple
val VoiceActive         = Color(0xFF8B9AFF)   // Glow blue
val VoiceProcessing     = Color(0xFF9BB5FF)   // Glass aura
val VoiceError          = Color(0xFFFF6B8A)   // Soft error (minimalist)

// ===== GRADIENT SYSTEM (Soft Blue-Purple) =====
val AIGradientStart     = Color(0xFF0A0B0F)   // Deep black-blue
val AIGradientMiddle    = Color(0xFF1A1D2B)   // Soft dark
val AIGradientEnd       = Color(0xFF2A2F42)   // Glass surface

// ===== ORBIT MOTION COLORS (Glass aura effects) =====
val OrbitColor1         = Color(0xFF4A5BCC)   // Deep soft blue
val OrbitColor2         = Color(0xFF6B73FF)   // Soft blue-purple
val OrbitColor3         = Color(0xFF8B9AFF)   // Glow blue
val OrbitColor4         = Color(0xFF9BB5FF)   // Glass aura
val OrbitColor5         = Color(0xFFB8C5FF)   // Light glass

// ===== PULSATING SPHERE COLORS =====
val SphereCore          = Color(0xFF6B73FF)   // Core soft blue-purple
val SphereGlow          = Color(0xFF8B9AFF)   // Glow layer
val SphereAura          = Color(0xFF9BB5FF)   // Glass aura
val SphereBreath        = Color(0xFF4A5BCC)   // Breathing deep

// ===== GLASS MORPHISM EFFECTS =====
val GlassBackground     = Color(0x1A2A2F42)   // Glass background
val GlassBorder         = Color(0x336B73FF)   // Glass border
val GlassGlow           = Color(0x4D8B9AFF)   // Glass glow

// ===== SEMANTIC COLORS (Minimalist) =====
val AISuccess           = Color(0xFF4DD0E1)   // Soft cyan
val AIError             = Color(0xFFFF6B8A)   // Soft error
val AIWarning           = Color(0xFFFFB74D)   // Soft warning
val AIInfo              = Color(0xFF64B5F6)   // Soft info

// ===== FLOATING WINDOW (AI Style) =====
val FloatingButtonColor     = AIPrimarySoft
val FloatingButtonPressed   = AIPrimaryGlow
val FloatingMenuBackground  = AISurfaceGlass

// ===== GOLDEN SPHERE THEME (Warm Golden-Orange) =====
// Inspired by golden orb with warm, energetic feel
// REVERSE GRADIENT: Vàng rực ngoài → Hồng/đào trong
val GoldenCore          = Color(0xFFFFB6C1)   // Light pink core (tâm)
val GoldenInner         = Color(0xFFFFA07A)   // Light salmon (trong)
val GoldenMiddle        = Color(0xFFFF8C00)   // Dark orange (giữa)
val GoldenOuter         = Color(0xFFFFD700)   // Bright golden (ngoài)
val GoldenAura          = Color(0xFFFFA500)   // Orange aura
val GoldenBreath        = Color(0xFFFF7F50)   // Coral breathing
val GoldenBorder        = Color(0xFFFF6347)   // Tomato border glow
val GoldenBackground    = Color(0xFFFFF8DC)   // Cream background
val GoldenSurface       = Color(0xFFFFFACD)   // Lemon chiffon surface

// ===== GOLDEN VOICE RESPONSIVE COLORS =====
val GoldenVoiceIdle     = Color(0xFFFFD700)   // Golden idle
val GoldenVoiceListening = Color(0xFFFFA500) // Orange listening
val GoldenVoiceActive   = Color(0xFFFF8C00)   // Deep orange active
val GoldenVoiceProcessing = Color(0xFFFF7F50) // Coral processing
val GoldenVoiceError    = Color(0xFFFF4500)   // Orange red error

// ===== GOLDEN ORBIT MOTION COLORS =====
val GoldenOrbit1        = Color(0xFFFFD700)   // Bright gold
val GoldenOrbit2        = Color(0xFFFFA500)   // Orange
val GoldenOrbit3        = Color(0xFFFF8C00)   // Dark orange
val GoldenOrbit4        = Color(0xFFFF7F50)   // Coral
val GoldenOrbit5        = Color(0xFFFF6347)   // Tomato

// ===== LEGACY COMPATIBILITY (for existing components) =====
val DarkPrimary         = AIPrimarySoft
val DarkBackground      = AIBackgroundDeep
val DarkSurface         = AISurfaceGlass
val DarkOnPrimary       = AITextPrimary
val DarkOnBackground    = AITextPrimary
val DarkOnSurface       = AITextPrimary
val DarkError           = AIError
val SuccessColor        = AISuccess
