package com.auto_fe.auto_fe.utils.common

import android.content.Context
import android.content.SharedPreferences

class SettingsManager(context: Context) {
    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREFS_NAME = "auto_fe_settings"
        private const val SETTING_PREFIX = "setting_"
        private const val DEFAULT_PREFIX = "default_setting_"
        private const val VOICE_SUPPORT_KEY = "voice_support"
    }

    fun isSupportSpeakEnabled(): Boolean {
        val value = getSettingValue(VOICE_SUPPORT_KEY)
        return when (value?.lowercase()) {
            "on", "true", "1", "yes" -> true
            "off", "false", "0", "no" -> false
            else -> true
        }
    }

    fun setSupportSpeakEnabled(enabled: Boolean) {
        setSettingValue(VOICE_SUPPORT_KEY, if (enabled) "on" else "off")
    }

    fun getSettingValue(settingKey: String, defaultValue: String? = null): String? {
        val key = "$SETTING_PREFIX$settingKey"
        return prefs.getString(key, defaultValue)
    }

    fun setSettingValue(settingKey: String, value: String) {
        val key = "$SETTING_PREFIX$settingKey"
        prefs.edit().putString(key, value).apply()
    }

    fun saveDefaultValues(settingsWithDefaults: Map<String, String?>) {
        val editor = prefs.edit()
        settingsWithDefaults.forEach { (settingKey, defaultValue) ->
            val key = "$DEFAULT_PREFIX$settingKey"
            if (defaultValue != null) {
                editor.putString(key, defaultValue)
            } else {
                editor.remove(key)
            }
        }
        editor.apply()
    }

    fun resetAllSettingsToDefault() {
        val editor = prefs.edit()
        val allPrefs = prefs.all
        
        allPrefs.forEach { (key, _) ->
            if (key.startsWith(DEFAULT_PREFIX)) {
                val settingKey = key.removePrefix(DEFAULT_PREFIX)
                val defaultValue = prefs.getString(key, null)
                val settingKeyWithPrefix = "$SETTING_PREFIX$settingKey"
                
                if (defaultValue != null) {
                    editor.putString(settingKeyWithPrefix, defaultValue)
                } else {
                    editor.remove(settingKeyWithPrefix)
                }
            }
        }
        editor.apply()
    }

    fun saveSettings(settings: Map<String, String>) {
        val editor = prefs.edit()
        settings.forEach { (settingKey, value) ->
            val key = "$SETTING_PREFIX$settingKey"
            editor.putString(key, value)
        }
        editor.apply()
    }
}
