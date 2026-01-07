package com.auto_fe.auto_fe.utils.nlp

import android.content.Context
import android.media.AudioManager
import android.net.wifi.WifiManager
import android.provider.Settings

/**
 * Utility functions cho Device Control Automation
 * Các hàm helper để lấy trạng thái và thông tin thiết bị
 */
object DeviceControlUtils {
    
    /**
     * Lấy trạng thái WiFi
     */
    fun isWifiEnabled(context: Context): Boolean {
        val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        return wifiManager.isWifiEnabled
    }
    
    /**
     * Lấy âm lượng hiện tại (0-100%)
     */
    fun getCurrentVolumePercentage(context: Context): Int {
        val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
        val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        return if (maxVolume > 0) {
            ((currentVolume.toFloat() / maxVolume) * 100).toInt()
        } else {
            0
        }
    }
    
    /**
     * Lấy độ sáng hiện tại (0-100%)
     */
    fun getCurrentBrightnessPercentage(context: Context): Int {
        val currentBrightness = Settings.System.getInt(
            context.contentResolver,
            Settings.System.SCREEN_BRIGHTNESS,
            0
        )
        return ((currentBrightness.toFloat() / 255) * 100).toInt()
    }
}

