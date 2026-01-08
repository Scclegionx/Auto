package com.auto_fe.auto_fe.automation.calendar

import android.content.Context
import android.content.Intent
import android.provider.CalendarContract
import android.provider.CalendarContract.Events
import android.util.Log
import android.widget.Toast

class CalendarAutomation(private val context: Context) {

    interface CalendarCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    fun addEvent(title: String, location: String, begin: Long, end: Long, callback: CalendarCallback) {
        try {
            Log.d("CalendarAutomation", "Creating event: $title at $location from $begin to $end")

            // Tạo sự kiện trực tiếp vào database calendar
            val values = android.content.ContentValues().apply {
                put(Events.TITLE, title)
                put(Events.EVENT_LOCATION, location)
                put(Events.DTSTART, begin)
                put(Events.DTEND, end)
                put(Events.CALENDAR_ID, 1) // Calendar ID mặc định
                put(Events.EVENT_TIMEZONE, java.util.TimeZone.getDefault().id)
                put(Events.ALL_DAY, 0) // Không phải cả ngày
            }

            val uri = context.contentResolver.insert(Events.CONTENT_URI, values)
            if (uri != null) {
                Log.d("CalendarAutomation", "Event created successfully: $uri")
                callback.onSuccess()
            } else {
                Log.e("CalendarAutomation", "Failed to create event")
                callback.onError("Dạ, con không thể tạo sự kiện ạ.")
            }

        } catch (e: Exception) {
            Log.e("CalendarAutomation", "Exception in addEvent: ${e.message}", e)
            callback.onError("Dạ, con không thể tạo sự kiện ạ.")
        }
    }

    fun createDefaultEvent(callback: CalendarCallback) {
        // Tạo sự kiện mặc định: Họp thứ 4 tới lúc 10h sáng
        val title = "Họp thứ 4"
        val location = "Văn phòng"
        
        // Tính thời gian: Thứ 4 tới lúc 10h sáng
        val calendar = java.util.Calendar.getInstance()
        calendar.set(java.util.Calendar.DAY_OF_WEEK, java.util.Calendar.WEDNESDAY)
        calendar.set(java.util.Calendar.HOUR_OF_DAY, 10)
        calendar.set(java.util.Calendar.MINUTE, 0)
        calendar.set(java.util.Calendar.SECOND, 0)
        calendar.set(java.util.Calendar.MILLISECOND, 0)
        
        // Nếu thứ 4 đã qua trong tuần này, chuyển sang tuần sau
        if (calendar.timeInMillis <= System.currentTimeMillis()) {
            calendar.add(java.util.Calendar.WEEK_OF_YEAR, 1)
        }
        
        val begin = calendar.timeInMillis
        val end = calendar.timeInMillis + (60 * 60 * 1000) // 1 giờ sau

        addEvent(title, location, begin, end, callback)
    }
}
