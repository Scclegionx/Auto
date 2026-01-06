package com.auto_fe.auto_fe.utils.common

import android.content.Context
import android.content.SharedPreferences

class SettingsManager(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREFS_NAME = "auto_fe_settings"
        private const val KEY_SUPPORT_SPEAK_ENABLED = "support_speak_enabled"
    }

    fun isSupportSpeakEnabled(): Boolean =
        prefs.getBoolean(KEY_SUPPORT_SPEAK_ENABLED, true)

    fun setSupportSpeakEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_SUPPORT_SPEAK_ENABLED, enabled).apply()
    }
}

