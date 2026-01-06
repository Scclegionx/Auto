package com.auto_fe.auto_fe.utils.be

import android.os.Handler
import android.os.Looper
import android.util.Log
import okhttp3.Authenticator
import okhttp3.Request
import okhttp3.Response
import okhttp3.Route

/**
 * OkHttp Authenticator để tự động xử lý token expiration
 * Khi nhận 401, sẽ trigger logout callback
 */
class TokenAuthenticator(
    private val sessionManager: SessionManager,
    private val onTokenExpired: () -> Unit
) : Authenticator {
    
    private val mainHandler = Handler(Looper.getMainLooper())
    
    override fun authenticate(route: Route?, response: Response): Request? {
        // Nếu đã retry 3 lần rồi, dừng lại
        if (responseCount(response) >= 3) {
            Log.d("TokenAuthenticator", "Max retries reached, giving up")
            triggerLogout()
            return null
        }
        
        // Response 401 = token hết hạn hoặc invalid
        if (response.code == 401) {
            Log.d("TokenAuthenticator", "401 Unauthorized - Token expired or invalid")
            
            // TODO: Nếu backend có API refresh token, implement ở đây
            // val newToken = refreshToken()
            // if (newToken != null) {
            //     return response.request.newBuilder()
            //         .header("Authorization", "Bearer $newToken")
            //         .build()
            // }
            
            // Hiện tại: Không có refresh token API → logout
            triggerLogout()
            return null
        }
        
        return null
    }
    
    /**
     * Đếm số lần retry
     */
    private fun responseCount(response: Response): Int {
        var result = 1
        var currentResponse = response.priorResponse
        while (currentResponse != null) {
            result++
            currentResponse = currentResponse.priorResponse
        }
        return result
    }
    
    /**
     * Trigger logout trên main thread
     */
    private fun triggerLogout() {
        Log.d("TokenAuthenticator", "Triggering logout due to authentication failure")
        sessionManager.clearSession()
        mainHandler.post {
            onTokenExpired()
        }
    }
}

