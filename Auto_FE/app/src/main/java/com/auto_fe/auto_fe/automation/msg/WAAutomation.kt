package com.auto_fe.auto_fe.automation.msg

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import com.auto_fe.auto_fe.automation.msg.SMSAutomation

class WAAutomation(private val context: Context) {
    
    interface WACallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    /**
     * Gửi tin nhắn qua WhatsApp sử dụng deeplink
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param message Nội dung tin nhắn
     */
    fun sendWA(receiver: String, message: String, callback: WACallback) {
        try {
            Log.d("WAAutomation", "sendWA called with receiver: $receiver, message: $message")
            
            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d("WAAutomation", "Receiver is phone number: $receiver")
                receiver
            } else {
                Log.d("WAAutomation", "Looking up phone number for contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d("WAAutomation", "Found phone number: $foundNumber")
                foundNumber
            }
            
            if (phoneNumber.isEmpty()) {
                Log.e("WAAutomation", "No phone number found for: $receiver")
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }
            
            Log.d("WAAutomation", "Using phone number: $phoneNumber")
            
            // Format số điện thoại cho WhatsApp deeplink
            val formattedPhoneNumber = formatPhoneNumberForWhatsApp(phoneNumber)
            Log.d("WAAutomation", "Formatted phone number for WhatsApp: $formattedPhoneNumber")
            
            // Tạo deeplink WhatsApp
            val whatsappUrl = "https://wa.me/$formattedPhoneNumber?text=${Uri.encode(message)}"
            Log.d("WAAutomation", "WhatsApp URL: $whatsappUrl")
            
            try {
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse(whatsappUrl)
                    setPackage("com.whatsapp")
                }
                
                if (intent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(intent)
                    Log.d("WAAutomation", "WhatsApp opened successfully")
                    callback.onSuccess()
                } else {
                    Log.e("WAAutomation", "WhatsApp not installed")
                    callback.onError("WhatsApp chưa được cài đặt")
                }
            } catch (e: Exception) {
                Log.e("WAAutomation", "Failed to open WhatsApp: ${e.message}")
                callback.onError("Không thể mở WhatsApp: ${e.message}")
            }
            
        } catch (e: Exception) {
            Log.e("WAAutomation", "Exception in sendWA: ${e.message}", e)
            callback.onError("Lỗi gửi WhatsApp: ${e.message}")
        }
    }
    
    /**
     * Kiểm tra xem chuỗi có phải là số điện thoại không
     */
    private fun isPhoneNumber(input: String): Boolean {
        return input.matches(Regex("^[+]?[0-9\\s\\-\\(\\)]+$"))
    }
    
    /**
     * Tìm số điện thoại từ tên liên hệ (sử dụng lại logic từ SMSAutomation)
     */
    private fun findPhoneNumberByName(contactName: String): String {
        Log.d("WAAutomation", "Searching for contact: $contactName")
        
        // Sử dụng SMSAutomation để tìm số điện thoại
        val smsAutomation = SMSAutomation(context)
        return smsAutomation.findPhoneNumberByName(contactName)
    }
    
    /**
     * Format số điện thoại cho WhatsApp deeplink
     * Chuyển từ "098 538 35 69" thành "+84985383569"
     */
    private fun formatPhoneNumberForWhatsApp(phoneNumber: String): String {
        // Bỏ tất cả khoảng cách, dấu gạch ngang, dấu ngoặc
        val cleanNumber = phoneNumber.replace(Regex("[\\s\\-\\(\\)]"), "")
        
        return when {
            // Nếu bắt đầu bằng +84, giữ nguyên
            cleanNumber.startsWith("+84") -> cleanNumber
            
            // Nếu bắt đầu bằng 84, thêm dấu +
            cleanNumber.startsWith("84") -> "+$cleanNumber"
            
            // Nếu bắt đầu bằng 0, thay thế bằng +84
            cleanNumber.startsWith("0") -> "+84${cleanNumber.substring(1)}"
            
            // Các trường hợp khác, thêm +84
            else -> "+84$cleanNumber"
        }
    }
}
