package com.auto_fe.auto_fe.automation.third_apps

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log

class YouTubeAutomation(private val context: Context) {
    
    interface YouTubeCallback {
        fun onSuccess()
        fun onError(error: String)
    }
    
    fun searchYouTube(query: String, callback: YouTubeCallback) {
        try {
            Log.d("YouTubeAutomation", "Searching YouTube for: $query")
            
            // Cách 1: Thử mở YouTube app trực tiếp với intent ACTION_VIEW
            val youtubeIntent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("https://www.youtube.com/results?search_query=${Uri.encode(query)}")
                setPackage("com.google.android.youtube")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            // Kiểm tra xem YouTube app có cài đặt không
            if (youtubeIntent.resolveActivity(context.packageManager) != null) {
                context.startActivity(youtubeIntent)
                Log.d("YouTubeAutomation", "YouTube app opened with search: $query")
                callback.onSuccess()
            } else {
                // Cách 2: Fallback - Mở YouTube trên web browser
                Log.w("YouTubeAutomation", "YouTube app not found, opening web browser")
                val webIntent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("https://www.youtube.com/results?search_query=${Uri.encode(query)}")
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                
                if (webIntent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(webIntent)
                    Log.d("YouTubeAutomation", "YouTube web opened with search: $query")
                    callback.onSuccess()
                } else {
                    Log.e("YouTubeAutomation", "No app found to handle YouTube search")
                    callback.onError("Không tìm thấy ứng dụng để mở YouTube")
                }
            }
            
        } catch (e: Exception) {
            Log.e("YouTubeAutomation", "Exception in searchYouTube: ${e.message}", e)
            callback.onError("Lỗi tìm kiếm YouTube: ${e.message}")
        }
    }
    
    fun searchDefault(callback: YouTubeCallback) {
        // Tìm kiếm mặc định: "nhạc sơn tùng MTP"
        searchYouTube("nhạc sơn tùng MTP", callback)
    }
}
