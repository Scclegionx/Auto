package com.auto_fe.auto_fe.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// Larger typography for better readability for elderly users
val Typography = Typography(
    // Title Large - For main headings
    titleLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Bold,
        fontSize = 28.sp,
        lineHeight = 36.sp,
        letterSpacing = 0.sp
    ),
    // Title Medium - For section headings
    titleMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Bold,
        fontSize = 23.sp,
        lineHeight = 32.sp,
        letterSpacing = 0.sp
    ),
    // Body Large - For main content text
    bodyLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 20.sp,
        lineHeight = 28.sp,
        letterSpacing = 0.25.sp
    ),
    // Body Medium - For secondary content
    bodyMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Normal,
        fontSize = 18.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.25.sp
    ),
    // Label Large - For buttons and important labels
    labelLarge = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Medium,
        fontSize = 18.sp,
        lineHeight = 24.sp,
        letterSpacing = 0.25.sp
    ),
    // Label Medium - For smaller buttons and labels
    labelMedium = TextStyle(
        fontFamily = FontFamily.Default,
        fontWeight = FontWeight.Medium,
        fontSize = 16.sp,
        lineHeight = 20.sp,
        letterSpacing = 0.25.sp
    )
)

// Font sizes cho các màn hình sau khi login (to hơn cho người cao tuổi)
object AppTextSize {
    val titleLarge = 32.sp      // Tiêu đề chính (tăng từ 28sp)
    val titleMedium = 26.sp     // Tiêu đề phụ (tăng từ 23sp)
    val titleSmall = 22.sp      // Tiêu đề nhỏ
    val bodyLarge = 22.sp       // Nội dung chính (tăng từ 20sp)
    val bodyMedium = 20.sp      // Nội dung phụ (tăng từ 18sp)
    val bodySmall = 18.sp       // Nội dung nhỏ
    val labelLarge = 20.sp      // Label button lớn (tăng từ 18sp)
    val labelMedium = 18.sp     // Label button vừa (tăng từ 16sp)
    val labelSmall = 16.sp      // Label nhỏ
}
