package com.auto_fe.auto_fe.automation.alarm

import android.content.Context
import android.content.Intent
import android.provider.AlarmClock
import android.util.Log
import com.auto_fe.auto_fe.utils.common.parseDateIso
import com.auto_fe.auto_fe.utils.common.parseTimeFromString
import org.json.JSONObject

class AlarmAutomation(private val context: Context) {

    companion object {
        private const val TAG = "AlarmAutomation"
    }

    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing alarm with entities: $entities")

        // Parse dữ liệu
        val time = entities.optString("TIME", "")
        val date = entities.optString("DATE", "")
        val message = entities.optString("MESSAGE", "Báo thức")

        // Validate Time
        val timeData = parseTimeFromString(time)
            ?: throw Exception("Tôi không hiểu thời gian báo thức. Vui lòng nói rõ giờ phút.")

        val (hour, minute) = timeData

        // Routing logic: Có ngày hay không có ngày
        return if (date.isNotEmpty()) {
            val dateData = parseDateIso(date) 
                ?: throw Exception("Định dạng ngày không hợp lệ.")
            
            createAlarmOnDate(
                year = dateData.first,
                month = dateData.second,
                day = dateData.third,
                hour = hour,
                minute = minute,
                message = message
            )
        } else {
            createAlarm(
                hour = hour,
                minute = minute,
                daysOfWeek = emptyList(),
                message = message
            )
        }
    }
    private fun createAlarm(
        hour: Int, 
        minute: Int, 
        daysOfWeek: List<Int>, 
        message: String
    ): String {
        return try {
            Log.d(TAG, "Creating alarm: $hour:$minute, days: $daysOfWeek")

            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_MESSAGE, message)
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                putExtra(AlarmClock.EXTRA_SKIP_UI, true) // Tự động set
                
                // Xử lý lặp lại ngày (nếu có)
                if (daysOfWeek.isNotEmpty()) {
                    putExtra(AlarmClock.EXTRA_DAYS, ArrayList(daysOfWeek))
                }
                
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }

            context.startActivity(intent)
            
            // Trả về thông báo thành công
            "Đã đặt báo thức lúc $hour:$minute với nội dung: $message"

        } catch (e: Exception) {
            Log.e(TAG, "Error creating alarm", e)
            throw Exception("Không thể mở ứng dụng đồng hồ: ${e.message}")
        }
    }

    private fun createAlarmOnDate(
        year: Int, month: Int, day: Int, 
        hour: Int, minute: Int, 
        message: String
    ): String {
        // Format ngày tháng để gắn vào message
        val dateSuffix = String.format("%02d/%02d/%04d", day, month, year)
        val enhancedMessage = if (message.isNotEmpty()) "$message ($dateSuffix)" else dateSuffix
        
        // Gọi lại hàm cốt lõi
        val resultMsg = createAlarm(hour, minute, emptyList(), enhancedMessage)
        
        // Override lại câu thông báo cho chi tiết hơn
        return "Đã đặt báo thức lúc $hour:$minute ngày $dateSuffix"
    }
}
