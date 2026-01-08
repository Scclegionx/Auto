package com.auto_fe.auto_fe.automation.msg

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import android.telephony.SmsManager
import com.auto_fe.auto_fe.base.ConfirmationRequirement
import com.auto_fe.auto_fe.utils.common.SettingsManager
import com.auto_fe.auto_fe.utils.nlp.ContactUtils
import org.json.JSONObject

/**
 * SMSAutomation - Refactored for CommandDispatcher
 */
class SMSAutomation(private val context: Context) {

    companion object {
        private const val TAG = "SMSAutomation"
    }

    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject, originalInput: String = ""): String {
        Log.d(TAG, "Executing SMS with entities: $entities")

        // Parse dữ liệu
        val receiver = entities.optString("RECEIVER", "")
        val message = entities.optString("MESSAGE", "")

        // Validate
        if (receiver.isEmpty()) {
            throw Exception("Dạ, con chưa nghe rõ tên người cần gửi tin nhắn ạ. Bác vui lòng nói lại nhé.")
        }
        if (message.isEmpty()) {
            throw Exception("Dạ, con chưa nghe rõ nội dung tin nhắn ạ. Bác vui lòng nói lại nhé.")
        }

        // Kiểm tra setting hỗ trợ nói
        val settingsManager = SettingsManager(context)
        val isSupportSpeakEnabled = settingsManager.isSupportSpeakEnabled()

        // Routing logic: Gửi trực tiếp hay mở hộp thoại soạn tin
        if (isSupportSpeakEnabled) {
            // Bật hỗ trợ nói: Cần xác nhận trước khi gửi
            val confirmationQuestion = "Dạ, có phải bác muốn $originalInput?"
            throw ConfirmationRequirement(
                originalInput = originalInput,
                confirmationQuestion = confirmationQuestion,
                onConfirmed = {
                    sendSMS(receiver, message)
                }
            )
        } else {
            // Tắt hỗ trợ nói: Mở hộp soạn tin
            return openSmsCompose(receiver, message)
        }
    }

    /**
     * Gửi SMS sử dụng SmsManager API
     */
    private fun sendSMS(receiver: String, message: String): String {
        return try {
            Log.d(TAG, "sendSMS called with receiver: $receiver, message: $message")

            val phoneNumber = if (ContactUtils.isPhoneNumber(receiver)) {
                Log.d(TAG, "Receiver is phone number: $receiver")
                receiver
            } else {
                Log.d(TAG, "Looking up phone number for contact: $receiver")
                val foundNumber = ContactUtils.findPhoneNumberByName(context, receiver)
                Log.d(TAG, "Found phone number: $foundNumber")
                foundNumber
            }

            if (phoneNumber.isEmpty()) {
                Log.e(TAG, "No phone number found for: $receiver")
                throw Exception("Dạ, trong danh bạ chưa có tên này ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gửi tin nhắn nhé.")
            }

            Log.d(TAG, "Using phone number: $phoneNumber")

            try {
                val smsManager = SmsManager.getDefault()
                smsManager.sendTextMessage(phoneNumber, null, message, null, null)
                Log.d(TAG, "SMS sent successfully via SmsManager")
                "Dạ, đã gửi tin nhắn đến $receiver ạ."
            } catch (e: Exception) {
                Log.e(TAG, "Failed to send SMS via SmsManager: ${e.message}")
                throw Exception("Dạ, con không thể gửi tin nhắn ạ.")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Exception in sendSMS: ${e.message}", e)
            throw Exception("Dạ, con không thể gửi tin nhắn ạ.")
        }
    }

    /**
     * Mở màn hình soạn SMS (không gửi tự động), điền sẵn số và nội dung
     */
    private fun openSmsCompose(receiver: String, message: String): String {
        return try {
            Log.d(TAG, "openSmsCompose called with receiver: $receiver, message: $message")

            val phoneNumber = if (ContactUtils.isPhoneNumber(receiver)) {
                receiver
            } else {
                ContactUtils.findPhoneNumberByName(context, receiver)
            }
            
            if (phoneNumber.isEmpty()) {
                throw Exception("Dạ, trong danh bạ chưa có tên này ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại lệnh gửi tin nhắn nhé.")
            }

            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("smsto:$phoneNumber")
                putExtra("sms_body", message)
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Opened SMS compose UI successfully")
                "Dạ, đã mở màn hình soạn tin nhắn. Bác kiểm tra lại nội dung và bấm gửi nhé."
            } else {
                Log.e(TAG, "No app available to handle SMS compose")
                throw Exception("Dạ, con không tìm thấy ứng dụng nhắn tin ạ.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception in openSmsCompose: ${e.message}", e)
            throw Exception("Dạ, con không thể mở màn hình soạn tin nhắn ạ.")
        }
    }
}
