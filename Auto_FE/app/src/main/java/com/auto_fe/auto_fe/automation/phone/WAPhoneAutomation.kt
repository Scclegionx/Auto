package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import com.auto_fe.auto_fe.utils.nlp.ContactUtils

class WAPhoneAutomation(private val context: Context) {
    
    interface WAPhoneCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    companion object {
        private const val TAG = "WAPhoneAutomation"
    }
    
    /**
     * Gọi điện qua WhatsApp sử dụng deeplink
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param callback Callback để nhận kết quả
     */
    fun callWA(receiver: String, callback: WAPhoneCallback) {
        try {
            Log.d(TAG, "callWA called with receiver: $receiver")
            
            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
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
                callback.onError("Dạ, trong danh bạ chưa có tên này ạ. Bác vui lòng xem hướng dẫn thêm liên hệ tự động, sau đó hãy thử lại nhé.")
                return
            }
            
            Log.d(TAG, "Using phone number: $phoneNumber")
            
            // Làm sạch số điện thoại (bỏ khoảng cách, dấu gạch ngang, dấu ngoặc)
            val cleanPhoneNumber = phoneNumber.replace(Regex("[\\s\\-\\(\\)]"), "")
            Log.d(TAG, "Clean phone number: $cleanPhoneNumber")
            
            // Tạo deeplink WhatsApp để gọi điện
            val whatsappCallUrl = "https://wa.me/$cleanPhoneNumber"
            Log.d(TAG, "WhatsApp call URL: $whatsappCallUrl")
            
            try {
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse(whatsappCallUrl)
                    setPackage("com.whatsapp")
                }
                
                if (intent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(intent)
                    Log.d(TAG, "WhatsApp call opened successfully")
                    callback.onSuccess()
                } else {
                    Log.e(TAG, "WhatsApp not installed")
                    callback.onError("Dạ, WhatsApp chưa được cài đặt ạ.")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to open WhatsApp call: ${e.message}")
                callback.onError("Dạ, con không thể mở cuộc gọi WhatsApp ạ.")
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Exception in callWA: ${e.message}", e)
            callback.onError("Dạ, con không thể gọi WhatsApp ạ.")
        }
    }
    
    /**
     * Kiểm tra xem WhatsApp có được cài đặt không
     */
    fun isWhatsAppInstalled(): Boolean {
        return try {
            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("https://wa.me/")
                setPackage("com.whatsapp")
            }
            intent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking WhatsApp installation: ${e.message}")
            false
        }
    }
    
    /**
     * Lấy thông tin chi tiết về khả năng WhatsApp
     */
    fun getWhatsAppInfo(): Map<String, Any> {
        return mapOf(
            "whatsapp_installed" to isWhatsAppInstalled(),
            "can_call" to isWhatsAppInstalled(),
            "can_send_message" to isWhatsAppInstalled(),
            "can_open_chat" to isWhatsAppInstalled()
        )
    }
}
