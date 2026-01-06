package com.auto_fe.auto_fe.utils.common

import android.util.Log

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

/**
 * Parse time từ string (ví dụ: "9:00", "9h00", "9 giờ 00")
 * @return Pair<Hour, Minute> hoặc null nếu không parse được
 */
fun parseTimeFromString(timeString: String): Pair<Int, Int>? {
    return try {
        val cleanTime = timeString.replace("giờ", ":").replace("h", ":").trim()
        
        // Tìm pattern HH:MM hoặc H:MM
        val timePattern = Regex("(\\d{1,2}):(\\d{2})")
        val match = timePattern.find(cleanTime)
        
        if (match != null) {
            val hour = match.groupValues[1].toInt()
            val minute = match.groupValues[2].toInt()
            
            if (hour in 0..23 && minute in 0..59) {
                return Pair(hour, minute)
            }
        }
        
        // Fallback: tìm số đầu tiên làm giờ, số thứ hai làm phút
        val numbers = Regex("\\d+").findAll(cleanTime).map { it.value.toInt() }.toList()
        if (numbers.size >= 2) {
            val hour = numbers[0]
            val minute = numbers[1]
            if (hour in 0..23 && minute in 0..59) {
                return Pair(hour, minute)
            }
        }
        
        null
    } catch (e: Exception) {
        Log.e("DateUtils", "Error parsing time: ${e.message}")
        null
    }
}

/**
 * Parse DATE theo định dạng ISO "YYYY-MM-DD" và trả về (year, month, day)
 * @return Triple<Year, Month, Day> hoặc null nếu không parse được
 * month là 1..12
 */
fun parseDateIso(dateString: String): Triple<Int, Int, Int>? {
    return try {
        val parts = dateString.trim().split("-")
        if (parts.size != 3) return null
        val year = parts[0].toInt()
        val month = parts[1].toInt()
        val day = parts[2].toInt()
        if (year in 1970..3000 && month in 1..12 && day in 1..31) {
            Triple(year, month, day)
        } else null
    } catch (e: Exception) {
        Log.e("DateUtils", "Error parsing date: ${e.message}")
        null
    }
}

