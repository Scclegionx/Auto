package com.auto_fe.auto_fe.automation.third_apps

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import org.json.JSONObject

class ChromeAutomation(private val context: Context) {
    
    companion object {
        private const val TAG = "ChromeAutomation"
    }
    
    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing Chrome search with entities: $entities")

        // Parse dữ liệu
        val query = entities.optString("QUERY", "")

        // Validate
        if (query.isEmpty()) {
            throw Exception("Dạ, con chưa nghe rõ từ khóa tìm kiếm ạ. Bác vui lòng nói lại nhé.")
        }

        // Routing logic: Tìm kiếm trên Chrome
        return searchChrome(query)
    }
    
    /**
     * Tìm kiếm trên Chrome
     * @param query Từ khóa tìm kiếm
     */
    private fun searchChrome(query: String): String {
        return try {
            Log.d(TAG, "Searching Chrome for: $query")
            
            // Tạo intent để mở Chrome với query tìm kiếm
            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("https://www.google.com/search?q=${Uri.encode(query)}")
                setPackage("com.android.chrome") // Chrome package name
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            
            // Kiểm tra xem Chrome app có cài đặt không
            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Chrome app opened with search: $query")
                "Dạ, đã mở Chrome và tìm kiếm: $query ạ."
            } else {
                // Fallback: Mở browser mặc định
                Log.w(TAG, "Chrome app not found, opening default browser")
                val fallbackIntent = Intent(Intent.ACTION_VIEW).apply {
                    data = Uri.parse("https://www.google.com/search?q=${Uri.encode(query)}")
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                
                if (fallbackIntent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(fallbackIntent)
                    Log.d(TAG, "Default browser opened with search: $query")
                    "Dạ, đã mở trình duyệt mặc định và tìm kiếm: $query ạ."
                } else {
                    Log.e(TAG, "No browser found to handle search")
                    throw Exception("Dạ, con không tìm thấy trình duyệt để tìm kiếm ạ.")
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Exception in searchChrome: ${e.message}", e)
            throw Exception("Dạ, con không thể mở trình duyệt để tìm kiếm ạ.")
        }
    }
}
