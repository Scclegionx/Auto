package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.database.Cursor
import android.Manifest
import android.content.pm.PackageManager
import android.util.Log
import android.speech.tts.TextToSpeech
import androidx.core.content.ContextCompat
import java.util.*

class PhoneAutomation(private val context: Context) {

    private var tts: TextToSpeech? = null

    interface PhoneCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    init {
        // Khởi tạo Text-to-Speech cho tiếng Việt
        initTTS()
    }

    /**
     * Gọi điện sử dụng Android Intents API
     * Thử nhiều phương pháp fallback để đảm bảo tương thích tối đa
     * @param receiver Tên người nhận hoặc số điện thoại
     */
    fun makeCall(receiver: String, callback: PhoneCallback) {
        try {
            Log.d("PhoneAutomation", "makeCall called with receiver: $receiver")

            //Tìm số điện thoại từ tên liên hệ hoặc sử dụng trực tiếp nếu là số
            val phoneNumber = if (isPhoneNumber(receiver)) {
                Log.d("PhoneAutomation", "Using direct phone number: $receiver")
                receiver
            } else {
                Log.d("PhoneAutomation", "Looking up contact: $receiver")
                val foundNumber = findPhoneNumberByName(receiver)
                Log.d("PhoneAutomation", "Found phone number: $foundNumber")
                foundNumber
            }

            if (phoneNumber.isEmpty()) {
                Log.e("PhoneAutomation", "No phone number found for: $receiver")
                val errorMessage = "Không tìm thấy số điện thoại cho $receiver"
                speakError(errorMessage)
                callback.onError(errorMessage)
                return
            }

            Log.d("PhoneAutomation", "Attempting to call: $phoneNumber")
            var success = false

            // Cách 0: ACTION_CALL - Gọi trực tiếp (nếu có permission CALL_PHONE)
            // Gọi ngay lập tức, không cần user nhấn nút Call
            // Cần permission nguy hiểm, có thể bị từ chối
//            if (ContextCompat.checkSelfPermission(context, Manifest.permission.CALL_PHONE)
//                == PackageManager.PERMISSION_GRANTED) {
//                try {
//                    val callIntent = Intent(Intent.ACTION_CALL).apply {
//                        data = Uri.parse("tel:$phoneNumber")
//                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
//                    }
//                    Log.d("PhoneAutomation", "Trying ACTION_CALL (direct call)...")
//                    if (callIntent.resolveActivity(context.packageManager) != null) {
//                        context.startActivity(callIntent)
//                        Log.d("PhoneAutomation", "ACTION_CALL successful - calling directly")
//                        success = true
//                    } else {
//                        Log.w("PhoneAutomation", "ACTION_CALL resolveActivity returned null")
//                    }
//                } catch (e: Exception) {
//                    Log.e("PhoneAutomation", "ACTION_CALL failed: ${e.message}")
//                }
//            } else {
//                Log.d("PhoneAutomation", "CALL_PHONE permission not granted, skipping direct call")
//            }

            // Cách 1: ACTION_DIAL với Uri.parse - Phương pháp chuẩn
            // Không cần permission, an toàn, chuẩn theo Android docs
            // User phải nhấn nút Call thủ công
            if (!success) {
                try {
                    val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    Log.d("PhoneAutomation", "Trying ACTION_DIAL with Uri.parse...")
                    if (dialIntent.resolveActivity(context.packageManager) != null) {
                        context.startActivity(dialIntent)
                        Log.d("PhoneAutomation", "ACTION_DIAL successful")
                        success = true
                    } else {
                        Log.w("PhoneAutomation", "ACTION_DIAL resolveActivity returned null")
                    }
                } catch (e: Exception) {
                    Log.e("PhoneAutomation", "ACTION_DIAL failed: ${e.message}")
                }
            }

            // Cách 2: ACTION_VIEW fallback - Phương pháp generic
            // Ưu điểm: Broad compatibility, có thể handle bởi nhiều app
            // Sử dụng khi: ACTION_DIAL không được support hoặc bị block
            if (!success) {
                try {
                    val viewIntent = Intent(Intent.ACTION_VIEW).apply {
                        data = Uri.parse("tel:$phoneNumber")
                        flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    }
                    Log.d("PhoneAutomation", "Trying ACTION_VIEW fallback...")
                    context.startActivity(viewIntent)
                    Log.d("PhoneAutomation", "ACTION_VIEW successful")
                    success = true
                } catch (e: Exception) {
                    Log.e("PhoneAutomation", "ACTION_VIEW failed: ${e.message}")
                }
            }

            // Cách 3: Intent chooser cuối cùng - Nuclear option
            // Ưu điểm: Luôn work, let user chọn app
            // Sử dụng khi: Tất cả methods khác fail
            if (!success) {
                try {
                    val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                        data = Uri.parse("tel:$phoneNumber")
                    }
                    val chooser = Intent.createChooser(dialIntent, "Chọn ứng dụng gọi điện")
                    chooser.flags = Intent.FLAG_ACTIVITY_NEW_TASK
                    Log.d("PhoneAutomation", "Trying Intent chooser (last resort)...")
                    context.startActivity(chooser)
                    Log.d("PhoneAutomation", "Intent chooser successful")
                    success = true
                } catch (e: Exception) {
                    Log.e("PhoneAutomation", "Intent chooser failed: ${e.message}")
                }
            }

            // Kết quả cuối cùng
            if (success) {
                callback.onSuccess()
            } else {
                Log.e("PhoneAutomation", "All dialing methods failed")
                val errorMessage = "Không tìm thấy ứng dụng gọi điện"
                speakError(errorMessage)
                callback.onError(errorMessage)
            }

        } catch (e: Exception) {
            Log.e("PhoneAutomation", "makeCall exception: ${e.message}", e)
            val errorMessage = "Lỗi gọi điện: ${e.message}"
            speakError(errorMessage)
            callback.onError(errorMessage)
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

    /**
     * Khởi tạo Text-to-Speech cho tiếng Việt
     */
    private fun initTTS() {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts?.setLanguage(Locale("vi", "VN"))
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w("PhoneAutomation", "Vietnamese language not supported, using default")
                    tts?.setLanguage(Locale.getDefault())
                } else {
                    Log.d("PhoneAutomation", "Vietnamese TTS initialized successfully")
                }
            } else {
                Log.e("PhoneAutomation", "TTS initialization failed")
            }
        }
    }

    /**
     * Nói lỗi bằng tiếng Việt
     */
    private fun speakError(errorMessage: String) {
        try {
            Log.d("PhoneAutomation", "Speaking error: $errorMessage")
            tts?.speak(errorMessage, TextToSpeech.QUEUE_FLUSH, null, "ERROR_SPEECH")
        } catch (e: Exception) {
            Log.e("PhoneAutomation", "Failed to speak error: ${e.message}")
        }
    }

    /**
     * Giải phóng tài nguyên TTS
     */
    fun release() {
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            Log.d("PhoneAutomation", "TTS resources released")
        } catch (e: Exception) {
            Log.e("PhoneAutomation", "Error releasing TTS: ${e.message}")
        }
    }
}