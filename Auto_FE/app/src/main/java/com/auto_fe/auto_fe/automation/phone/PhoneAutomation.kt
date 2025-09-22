package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.database.Cursor
import android.Manifest
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat

class PhoneAutomation(private val context: Context) {
    
    interface PhoneCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    /**
     * Gọi điện sử dụng Android Intents API
     * @param receiver Tên người nhận hoặc số điện thoại
     */
    fun makeCall(receiver: String, callback: PhoneCallback) {
        try {
            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                receiver
            } else {
                findPhoneNumberByName(receiver)
            }
            
            if (phoneNumber.isEmpty()) {
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }
            
            // Sử dụng Intent.ACTION_DIAL để mở dialer (không cần quyền CALL_PHONE)
            val intent = Intent(Intent.ACTION_DIAL).apply {
                data = Uri.parse("tel:$phoneNumber")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            // Kiểm tra xem có app nào có thể xử lý intent này không
            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                callback.onSuccess()
            } else {
                callback.onError("Không tìm thấy ứng dụng gọi điện")
            }
            
        } catch (e: Exception) {
            callback.onError("Lỗi gọi điện: ${e.message}")
        }
    }
    
    /**
     * Gọi điện trực tiếp (cần quyền CALL_PHONE)
     */
    fun makeDirectCall(phoneNumber: String, callback: PhoneCallback) {
        // Kiểm tra quyền gọi điện
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CALL_PHONE) 
            != PackageManager.PERMISSION_GRANTED) {
            callback.onError("Không có quyền gọi điện")
            return
        }
        
        try {
            val intent = Intent(Intent.ACTION_CALL).apply {
                data = Uri.parse("tel:$phoneNumber")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            context.startActivity(intent)
            callback.onSuccess()
        } catch (e: Exception) {
            callback.onError("Lỗi gọi điện: ${e.message}")
        }
    }
    
    /**
     * Kiểm tra xem chuỗi có phải là số điện thoại không
     */
    private fun isPhoneNumber(input: String): Boolean {
        return input.matches(Regex("^[+]?[0-9\\s\\-\\(\\)]+$"))
    }
    
    /**
     * Tìm số điện thoại từ tên liên hệ
     */
    private fun findPhoneNumberByName(contactName: String): String {
        // Kiểm tra quyền đọc danh bạ
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) 
            != PackageManager.PERMISSION_GRANTED) {
            return ""
        }
        
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )
        
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")
        
        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )
        
        cursor?.use {
            if (it.moveToFirst()) {
                val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                return phoneNumber
            }
        }
        
        return ""
    }
}
