package com.auto_fe.auto_fe.automation.msg

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.database.Cursor
import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import androidx.core.content.ContextCompat

class SMSAutomation(private val context: Context) {
    
    interface SMSCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    /**
     * Gửi SMS sử dụng Android Intents API
     * @param receiver Tên người nhận hoặc số điện thoại
     * @param message Nội dung tin nhắn
     */
    fun sendSMS(receiver: String, message: String, callback: SMSCallback) {
        try {
            Log.d("SMSAutomation", "sendSMS called with receiver: $receiver, message: $message")
            
            // Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d("SMSAutomation", "Receiver is phone number: $receiver")
                receiver
            } else {
                Log.d("SMSAutomation", "Looking up phone number for contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d("SMSAutomation", "Found phone number: $foundNumber")
                foundNumber
            }
            
            if (phoneNumber.isEmpty()) {
                Log.e("SMSAutomation", "No phone number found for: $receiver")
                callback.onError("Không tìm thấy số điện thoại cho: $receiver")
                return
            }
            
            Log.d("SMSAutomation", "Using phone number: $phoneNumber")
            
            // Thử nhiều cách khác nhau để gửi SMS
            var intent: Intent? = null
            var canHandle = false
            
            // Cách 1: Sử dụng ACTION_SENDTO với smsto:
            intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("smsto:$phoneNumber")
                putExtra("sms_body", message)
            }
            canHandle = intent.resolveActivity(context.packageManager) != null
            Log.d("SMSAutomation", "Method 1 (smsto:) - Can handle: $canHandle")
            
            // Cách 2: Sử dụng ACTION_SENDTO với sms:
            if (!canHandle) {
                intent = Intent(Intent.ACTION_SENDTO).apply {
                    data = Uri.parse("sms:$phoneNumber")
                    putExtra("sms_body", message)
                }
                canHandle = intent.resolveActivity(context.packageManager) != null
                Log.d("SMSAutomation", "Method 2 (sms:) - Can handle: $canHandle")
            }
            
            // Cách 3: Sử dụng ACTION_SEND với MIME type
            if (!canHandle) {
                intent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, message)
                    putExtra("address", phoneNumber)
                }
                canHandle = intent.resolveActivity(context.packageManager) != null
                Log.d("SMSAutomation", "Method 3 (ACTION_SEND) - Can handle: $canHandle")
            }
            
            // Cách 4: Sử dụng ACTION_VIEW với sms:
            if (!canHandle) {
                intent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("sms:$phoneNumber")
                    putExtra("sms_body", message)
                }
                canHandle = intent.resolveActivity(context.packageManager) != null
                Log.d("SMSAutomation", "Method 4 (ACTION_VIEW) - Can handle: $canHandle")
            }
            
            Log.d("SMSAutomation", "Final intent: $intent")
            Log.d("SMSAutomation", "Intent data: ${intent?.data}")
            Log.d("SMSAutomation", "Intent extras: ${intent?.extras}")
            
            if (canHandle && intent != null) {
                intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                Log.d("SMSAutomation", "Starting SMS activity...")
                context.startActivity(intent)
                Log.d("SMSAutomation", "SMS activity started successfully")
                callback.onSuccess()
            } else {
                Log.e("SMSAutomation", "No app can handle any SMS intent")
                // Thử mở danh sách apps có thể gửi SMS
                val smsIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_TEXT, message)
                }
                val chooser = Intent.createChooser(smsIntent, "Chọn ứng dụng để gửi tin nhắn")
                chooser.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                context.startActivity(chooser)
                callback.onSuccess()
            }
            
        } catch (e: Exception) {
            Log.e("SMSAutomation", "Exception in sendSMS: ${e.message}", e)
            callback.onError("Lỗi gửi SMS: ${e.message}")
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
        Log.d("SMSAutomation", "Searching for contact: $contactName")
        
        // Kiểm tra quyền đọc danh bạ
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) 
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("SMSAutomation", "READ_CONTACTS permission not granted")
            return ""
        }
        
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )
        
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")
        
        Log.d("SMSAutomation", "Querying contacts with selection: $selection, args: ${selectionArgs.joinToString()}")
        
        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )
        
        cursor?.use {
            Log.d("SMSAutomation", "Found ${it.count} contacts")
            if (it.moveToFirst()) {
                val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                Log.d("SMSAutomation", "Found contact: $displayName -> $phoneNumber")
                return phoneNumber
            }
        }
        
        Log.d("SMSAutomation", "No contact found for: $contactName")
        return ""
    }
}
