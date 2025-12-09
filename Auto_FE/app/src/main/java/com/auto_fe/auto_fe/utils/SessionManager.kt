package com.auto_fe.auto_fe.utils

import android.content.Context
import android.content.SharedPreferences

class SessionManager(context: Context) {
    private val prefs: SharedPreferences = 
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREFS_NAME = "auto_fe_session"
        private const val KEY_ACCESS_TOKEN = "access_token"
        private const val KEY_REFRESH_TOKEN = "refresh_token"
        private const val KEY_USER_EMAIL = "user_email"
        private const val KEY_USER_NAME = "user_name"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_USER_AVATAR = "user_avatar"
        private const val KEY_USER_ROLE = "user_role" //  Add role
        private const val KEY_IS_LOGGED_IN = "is_logged_in"
        // Remember credentials
        private const val KEY_REMEMBER_ME = "remember_me"
        private const val KEY_SAVED_EMAIL = "saved_email"
        private const val KEY_SAVED_PASSWORD = "saved_password"
    }

    /**
     * Lưu thông tin đăng nhập
     */
    fun saveLoginSession(
        accessToken: String,
        refreshToken: String? = null,
        userEmail: String? = null,
        userName: String? = null,
        userId: Long? = null,
        userAvatar: String? = null,
        userRole: String? = null // Add role parameter
    ) {
        prefs.edit().apply {
            putString(KEY_ACCESS_TOKEN, accessToken)
            refreshToken?.let { putString(KEY_REFRESH_TOKEN, it) }
            userEmail?.let { putString(KEY_USER_EMAIL, it) }
            userName?.let { putString(KEY_USER_NAME, it) }
            userId?.let { putLong(KEY_USER_ID, it) }
            userAvatar?.let { putString(KEY_USER_AVATAR, it) }
            userRole?.let { putString(KEY_USER_ROLE, it) } // Save role
            putBoolean(KEY_IS_LOGGED_IN, true)
            apply()
        }
    }
    
    /**
     * Cập nhật avatar
     */
    fun updateUserAvatar(avatar: String?) {
        prefs.edit().apply {
            if (avatar != null) {
                putString(KEY_USER_AVATAR, avatar)
            } else {
                remove(KEY_USER_AVATAR)
            }
            apply()
        }
    }

    /**
     * Lấy access token
     */
    fun getAccessToken(): String? {
        return prefs.getString(KEY_ACCESS_TOKEN, null)
    }

    /**
     * Lấy refresh token
     */
    fun getRefreshToken(): String? {
        return prefs.getString(KEY_REFRESH_TOKEN, null)
    }

    /**
     * Lấy email người dùng
     */
    fun getUserEmail(): String? {
        return prefs.getString(KEY_USER_EMAIL, null)
    }

    /**
     * Lấy tên người dùng
     */
    fun getUserName(): String? {
        return prefs.getString(KEY_USER_NAME, null)
    }

    /**
     * Lấy user ID
     */
    fun getUserId(): Long? {
        val id = prefs.getLong(KEY_USER_ID, -1L)
        return if (id != -1L) id else null
    }
    
    /**
     * Lấy avatar người dùng
     */
    fun getUserAvatar(): String? {
        return prefs.getString(KEY_USER_AVATAR, null)
    }
    
    /**
     *  Lấy role người dùng (ELDER, SUPERVISOR, USER)
     */
    fun getUserRole(): String? {
        return prefs.getString(KEY_USER_ROLE, null)
    }

    /**
     * Kiểm tra trạng thái đăng nhập
     */
    fun isLoggedIn(): Boolean {
        return prefs.getBoolean(KEY_IS_LOGGED_IN, false) && 
               !getAccessToken().isNullOrEmpty()
    }

    /**
     * Xóa session (logout)
     */
    fun clearSession() {
        prefs.edit().apply {
            remove(KEY_ACCESS_TOKEN)
            remove(KEY_REFRESH_TOKEN)
            remove(KEY_USER_EMAIL)
            remove(KEY_USER_NAME)
            remove(KEY_USER_ID)
            remove(KEY_USER_AVATAR)
            remove(KEY_USER_ROLE) //  Clear role
            putBoolean(KEY_IS_LOGGED_IN, false)
            apply()
        }
    }

    /**
     * Cập nhật access token (dùng khi refresh token)
     */
    fun updateAccessToken(newAccessToken: String) {
        prefs.edit().putString(KEY_ACCESS_TOKEN, newAccessToken).apply()
    }
    
    // ==================== Remember Credentials ====================
    
    /**
     * Lưu credentials để ghi nhớ đăng nhập
     */
    fun saveRememberedCredentials(email: String, password: String) {
        prefs.edit().apply {
            putBoolean(KEY_REMEMBER_ME, true)
            putString(KEY_SAVED_EMAIL, email)
            // Simple encoding (Base64) - không an toàn 100% nhưng đủ cho demo
            // Nếu cần bảo mật cao hơn, dùng Android Keystore
            val encodedPassword = android.util.Base64.encodeToString(
                password.toByteArray(), 
                android.util.Base64.DEFAULT
            )
            putString(KEY_SAVED_PASSWORD, encodedPassword)
            apply()
        }
    }
    
    /**
     * Lấy email đã lưu
     */
    fun getSavedEmail(): String? {
        return if (isRememberMeEnabled()) {
            prefs.getString(KEY_SAVED_EMAIL, null)
        } else null
    }
    
    /**
     * Lấy password đã lưu (decoded)
     */
    fun getSavedPassword(): String? {
        return if (isRememberMeEnabled()) {
            val encoded = prefs.getString(KEY_SAVED_PASSWORD, null)
            encoded?.let {
                try {
                    String(android.util.Base64.decode(it, android.util.Base64.DEFAULT))
                } catch (e: Exception) {
                    null
                }
            }
        } else null
    }
    
    /**
     * Kiểm tra xem có bật Remember Me không
     */
    fun isRememberMeEnabled(): Boolean {
        return prefs.getBoolean(KEY_REMEMBER_ME, false)
    }
    
    /**
     * Xóa thông tin Remember Me
     */
    fun clearRememberedCredentials() {
        prefs.edit().apply {
            remove(KEY_REMEMBER_ME)
            remove(KEY_SAVED_EMAIL)
            remove(KEY_SAVED_PASSWORD)
            apply()
        }
    }
}
