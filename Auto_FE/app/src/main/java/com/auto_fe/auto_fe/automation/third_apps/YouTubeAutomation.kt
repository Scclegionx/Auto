package com.auto_fe.auto_fe.automation.third_apps

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import org.json.JSONObject

class YouTubeAutomation(private val context: Context) {
    
    companion object {
        private const val TAG = "YouTubeAutomation"
    }
    
    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing YouTube search with entities: $entities")

        // Parse dữ liệu
        val query = entities.optString("QUERY", "")

        // Validate
        if (query.isEmpty()) {
            throw Exception("Cần chỉ định từ khóa tìm kiếm")
        }

        // Routing logic: Tìm kiếm trên YouTube
        return searchYouTube(query)
    }
    
    /**
     * Tìm kiếm trên YouTube
     * @param query Từ khóa tìm kiếm
     */
    private fun searchYouTube(query: String): String {
        return try {
            Log.d(TAG, "Searching YouTube for: $query")
            
            // Cách 1: Thử mở YouTube app trực tiếp với intent ACTION_VIEW
            val youtubeIntent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("https://www.youtube.com/results?search_query=${Uri.encode(query)}")
                setPackage("com.google.android.youtube")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            // Kiểm tra xem YouTube app có cài đặt không
            if (youtubeIntent.resolveActivity(context.packageManager) != null) {
                context.startActivity(youtubeIntent)
                Log.d(TAG, "YouTube app opened with search: $query")
                "Đã mở YouTube và tìm kiếm: $query"
            } else {
                // Cách 2: Fallback - Mở YouTube trên web browser
                Log.w(TAG, "YouTube app not found, opening web browser")
                val webIntent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("https://www.youtube.com/results?search_query=${Uri.encode(query)}")
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                
                if (webIntent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(webIntent)
                    Log.d(TAG, "YouTube web opened with search: $query")
                    "Đã mở YouTube trên trình duyệt và tìm kiếm: $query"
                } else {
                    Log.e(TAG, "No app found to handle YouTube search")
                    throw Exception("Không tìm thấy ứng dụng để mở YouTube")
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Exception in searchYouTube: ${e.message}", e)
            throw Exception("Lỗi tìm kiếm YouTube: ${e.message}")
        }
    }
}
