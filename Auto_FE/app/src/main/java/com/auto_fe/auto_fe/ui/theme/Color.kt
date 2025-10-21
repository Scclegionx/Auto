package com.auto_fe.auto_fe.ui.theme

import androidx.compose.ui.graphics.Color

// ===== LIGHT THEME (Warm Minimal) =====
val LightPrimary        = Color(0xFFB07C4A)   // amber ấm (dùng cho nút phụ)
val LightBackground     = Color(0xFFF5F3EF)   // xám ngà ấm
val LightSurface        = Color(0xFFFFFFFF)   // trắng mềm
val LightOnPrimary      = Color(0xFFFFFFFF)
val LightOnBackground   = Color(0xFF1F1B16)
val LightOnSurface      = Color(0xFF2A2620)

// ===== DARK THEME — PURE DARK (Thuần đen, ít chói) =====
val DarkPrimary         = Color(0xFFE0A069)   // Amber ấm chỉ cho CTA
val DarkBackground      = Color(0xFF0A0E14)   // Đen xanh rất sâu
val DarkSurface         = Color(0xFF1A1F2E)   // Xanh đậm
val DarkOnPrimary       = Color(0xFF0A0E14)   // Đen trên amber
val DarkOnBackground    = Color(0xFFE8F4FD)   // Trắng xanh nhẹ
val DarkOnSurface       = Color(0xFFE8F4FD)   // Trắng xanh nhẹ

// ===== VOICE STATUS (Tông ấm nhất quán) =====
val VoiceLowColor       = Color(0xFFE9C489)   // amber mềm
val VoiceMediumColor    = Color(0xFFD4936A)   // cam ấm
val VoiceHighColor      = Color(0xFFC26A5E)   // coral ấm

// ===== GRADIENT BACKGROUND =====
val GradientStart       = Color(0xFFF7F5F2)   // ấm, có chiều sâu
val GradientEnd         = Color(0xFFEDE7E1)

val DarkGradientStart   = Color(0xFF0A0B0D)   // Đen sâu hơn
val DarkGradientEnd     = Color(0xFF171A1F)   // Xanh đậm hơn

// ===== WAVE / AURA (Tông ấm cho light, giữ xanh cho dark) =====
val WaveColor1          = Color(0xFFE5DAD0)   // xám – slate ấm
val WaveColor2          = Color(0xFFD9C9BC)   // KHÔNG xanh lam
val WaveColor3          = Color(0xFFC7B2A1)
val WaveColor4          = Color(0xFFB59C8B)
val WaveColor5          = Color(0xFFA18370)

val DarkWaveColor1      = Color(0xFF6A7BC4)   // Xanh tím sáng
val DarkWaveColor2      = Color(0xFF5A6BC0)   // Xanh tím
val DarkWaveColor3      = Color(0xFF4A5AB8)   // Xanh tím đậm
val DarkWaveColor4      = Color(0xFF3F51B5)   // Xanh tím sâu
val DarkWaveColor5      = Color(0xFF3949AB)   // Xanh tím rất đậm

// ===== SEMANTIC (Tông xanh nhất quán) =====
val SuccessColor        = Color(0xFF4CAF50)   // Xanh lá
val ErrorColor          = Color(0xFFF44336)   // Đỏ (giữ nguyên cho cảnh báo)
val DarkError           = Color(0xFFFF6B6B)   // Đỏ sáng cho dark theme
val WarningColor        = Color(0xFFFF9800)   // Cam (giữ nguyên cho cảnh báo)
val InfoColor           = Color(0xFF2196F3)   // Xanh dương

// ===== FLOATING WINDOW =====
val FloatingButtonColor     = DarkPrimary
val FloatingButtonPressed   = Color(0xFF29B6F6)
val FloatingMenuBackground  = DarkSurface

// ===== HIGH CONTRAST (khi cần) =====
val HighContrastText    = Color(0xFFFFFFFF)
val HighContrastBackground = Color(0xFF000000)
