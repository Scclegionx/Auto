package com.auto_fe.auto_fe.automation.alarm

import android.app.AlarmManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.provider.AlarmClock
import android.util.Log
import android.widget.Toast
import java.util.Calendar

class AlarmAutomation(private val context: Context) {

    interface AlarmCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    fun createAlarm(hour: Int, minute: Int, daysOfWeek: List<Int>, message: String, callback: AlarmCallback) {
        try {
            Log.d("AlarmAutomation", "Creating alarm: $hour:$minute, days: $daysOfWeek, message: $message")

            // Sử dụng cách đơn giản như bạn đề xuất
            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_MESSAGE, message)
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d("AlarmAutomation", "Alarm created successfully")
                callback.onSuccess()
            } else {
                Log.e("AlarmAutomation", "No app found to handle alarm intent")
                callback.onError("Không tìm thấy ứng dụng đồng hồ để tạo báo thức")
            }

        } catch (e: Exception) {
            Log.e("AlarmAutomation", "Exception in createAlarm: ${e.message}", e)
            callback.onError("Lỗi tạo báo thức: ${e.message}")
        }
    }

    fun createDefaultAlarm(callback: AlarmCallback) {
        // Tạo alarm mặc định: 9h sáng thứ 2 hàng tuần
        val hour = 9
        val minute = 0
        val daysOfWeek = listOf(2) // Thứ 2 (Calendar.MONDAY = 2)
        val message = "Báo thức hàng tuần"

        // Chỉ sử dụng intent để tạo alarm trong app Clock
        createAlarm(hour, minute, daysOfWeek, message, callback)
    }

}
