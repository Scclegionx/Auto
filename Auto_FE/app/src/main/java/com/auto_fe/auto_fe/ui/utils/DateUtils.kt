package com.auto_fe.auto_fe.ui.utils

/**
 * Helper function: Format daysOfWeek string to Vietnamese
 * Input: "1111111" = all days, "1111100" = T2-T6, etc.
 * Index: 0=T2, 1=T3, 2=T4, 3=T5, 4=T6, 5=T7, 6=CN
 */
fun formatDaysOfWeek(daysOfWeek: String): String {
    if (daysOfWeek.length != 7) return daysOfWeek
    
    // Check if all days
    if (daysOfWeek == "1111111") return "Hàng ngày"
    
    val dayNames = listOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
    val activeDays = daysOfWeek.mapIndexed { index, char ->
        if (char == '1') dayNames[index] else null
    }.filterNotNull()
    
    if (activeDays.isEmpty()) return "Không có ngày"
    
    // Check for weekdays (Mon-Fri: T2-T6)
    if (daysOfWeek == "1111100") return "Thứ 2 - Thứ 6"
    
    // Check for weekend (Sat-Sun: T7-CN)
    if (daysOfWeek == "0000011") return "Cuối tuần"
    
    return activeDays.joinToString(", ")
}
