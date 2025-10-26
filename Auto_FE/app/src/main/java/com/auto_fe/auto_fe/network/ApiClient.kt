package com.auto_fe.auto_fe.network

import com.auto_fe.auto_fe.utils.SessionManager
import com.auto_fe.auto_fe.utils.TokenAuthenticator
import okhttp3.OkHttpClient
import java.util.concurrent.TimeUnit

/**
 * Singleton để quản lý OkHttpClient với TokenAuthenticator
 */
object ApiClient {
    private var client: OkHttpClient? = null
    private var authenticator: TokenAuthenticator? = null
    
    /**
     * Khởi tạo OkHttpClient với authenticator
     * Gọi method này từ MainActivity khi app start
     */
    fun initialize(sessionManager: SessionManager, onTokenExpired: () -> Unit) {
        authenticator = TokenAuthenticator(sessionManager, onTokenExpired)
        
        client = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .authenticator(authenticator!!)
            .build()
    }
    
    /**
     * Lấy OkHttpClient instance
     * @throws IllegalStateException nếu chưa initialize
     */
    fun getClient(): OkHttpClient {
        return client ?: throw IllegalStateException(
            "ApiClient chưa được initialize. Gọi ApiClient.initialize() trước."
        )
    }
    
    /**
     * Reset client (dùng khi cần thay đổi config)
     */
    fun reset() {
        client = null
        authenticator = null
    }
}
