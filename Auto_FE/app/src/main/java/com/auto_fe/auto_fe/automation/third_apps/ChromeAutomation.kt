package com.auto_fe.auto_fe.automation.third_apps

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log

class ChromeAutomation(private val context: Context) {
    
    interface ChromeCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    fun searchChrome(query: String, callback: ChromeCallback) {
        try {
            Log.d("ChromeAutomation", "Searching Chrome for: $query")
            
            // Tạo intent để mở Chrome với query tìm kiếm
            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("https://www.google.com/search?q=${Uri.encode(query)}")
                setPackage("com.android.chrome") // Chrome package name
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            // Kiểm tra xem Chrome app có cài đặt không
            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d("ChromeAutomation", "Chrome app opened with search: $query")
                callback.onSuccess()
            } else {
                // Fallback: Mở browser mặc định
                Log.w("ChromeAutomation", "Chrome app not found, opening default browser")
                val fallbackIntent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("https://www.google.com/search?q=${Uri.encode(query)}")
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                
                if (fallbackIntent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(fallbackIntent)
                    Log.d("ChromeAutomation", "Default browser opened with search: $query")
                    callback.onSuccess()
                } else {
                    Log.e("ChromeAutomation", "No browser found to handle search")
                    callback.onError("Không tìm thấy trình duyệt để tìm kiếm")
                }
            }
            
        } catch (e: Exception) {
            Log.e("ChromeAutomation", "Exception in searchChrome: ${e.message}", e)
            callback.onError("Lỗi tìm kiếm Chrome: ${e.message}")
        }
    }
    
    fun searchDefault(callback: ChromeCallback) {
        // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
        searchChrome("nhạc sơn tùng MTP", callback)
    }
}
