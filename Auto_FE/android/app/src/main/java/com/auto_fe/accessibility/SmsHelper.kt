package com.auto_fe.accessibility

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.util.Log
import android.widget.Toast
import org.json.JSONObject

class SmsHelper(private val context: Context) {
    
    companion object {
        private const val TAG = "SmsHelper"
    }
    
    fun sendMessage(command: String, entities: String, values: String) {
        Log.d(TAG, "sendMessage called: command=$command, entities=$entities, values=$values")
        
        try {
            when (command) {
                "send-mes" -> {
                    val entitiesJson = JSONObject(entities)
                    val valuesJson = JSONObject(values)
                    
                    val contactName = entitiesJson.optString("ent", "")
                    val messageText = valuesJson.optString("val", "")
                    
                    if (contactName.isNotEmpty() && messageText.isNotEmpty()) {
                        sendSmsToContact(contactName, messageText)
                    } else {
                        Log.e(TAG, "Invalid entities or values")
                        Toast.makeText(context, "Dữ liệu không hợp lệ", Toast.LENGTH_SHORT).show()
                    }
                }
                else -> {
                    Log.e(TAG, "Unknown command: $command")
                    Toast.makeText(context, "Lệnh không được hỗ trợ", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendMessage: ${e.message}")
            Toast.makeText(context, "Lỗi: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun sendSmsToContact(contactName: String, messageText: String) {
        Log.d(TAG, "Sending SMS to $contactName: $messageText")
        
        try {
            // Lấy số điện thoại từ tên contact
            val phoneNumber = getPhoneNumberFromContact(contactName)
            
                         if (phoneNumber.isNotEmpty()) {
                 // Sử dụng Android Intent API để gửi SMS
                 val intent = Intent(Intent.ACTION_SENDTO).apply {
                     data = Uri.parse("sms:$phoneNumber") // Sử dụng sms: thay vì smsto:
                     putExtra("sms_body", messageText)
                 }
                 
                 try {
                     intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                     context.startActivity(intent)
                     Toast.makeText(context, "Mở app tin nhắn để gửi cho $contactName", Toast.LENGTH_LONG).show()
                     Log.d(TAG, "Đã mở app tin nhắn cho số: $phoneNumber")
                 } catch (e: Exception) {
                     Log.e(TAG, "Không thể mở app tin nhắn: ${e.message}")
                     Toast.makeText(context, "Không thể mở app tin nhắn", Toast.LENGTH_SHORT).show()
                 }
            } else {
                Toast.makeText(context, "Không tìm thấy số điện thoại cho $contactName", Toast.LENGTH_SHORT).show()
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in sendSmsToContact: ${e.message}")
            Toast.makeText(context, "Lỗi gửi tin nhắn: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun getPhoneNumberFromContact(contactName: String): String {
        Log.d(TAG, "Tìm kiếm contact: $contactName")
        
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.CONTACT_ID,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )
        
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")
        
        context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                val phoneNumber = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                val displayName = cursor.getString(cursor.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                Log.d(TAG, "Tìm thấy contact: $displayName - $phoneNumber")
                return phoneNumber
            }
        }
        
        Log.e(TAG, "Không tìm thấy contact: $contactName")
        return ""
    }
} 