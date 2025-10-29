package com.auto_fe.auto_fe.automation.device

import android.content.Context
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.wifi.WifiManager
import android.provider.Settings
import android.util.Log
import android.view.WindowManager

class ControlDeviceAutomation(private val context: Context) {

    interface DeviceCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private val wifiManager = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    private val windowManager = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager

    companion object {
        private const val TAG = "ControlDeviceAutomation"
    }

    // ========== WIFI CONTROL ==========

    /**
     * Bật WiFi
     */
    fun enableWifi(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Enabling WiFi")
            
            if (!wifiManager.isWifiEnabled) {
                wifiManager.isWifiEnabled = true
                Log.d(TAG, "WiFi enabled successfully")
                callback.onSuccess()
            } else {
                Log.d(TAG, "WiFi is already enabled")
                callback.onSuccess()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error enabling WiFi: ${e.message}", e)
            callback.onError("Lỗi bật WiFi: ${e.message}")
        }
    }

    /**
     * Tắt WiFi
     */
    fun disableWifi(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Disabling WiFi")
            
            if (wifiManager.isWifiEnabled) {
                wifiManager.isWifiEnabled = false
                Log.d(TAG, "WiFi disabled successfully")
                callback.onSuccess()
            } else {
                Log.d(TAG, "WiFi is already disabled")
                callback.onSuccess()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error disabling WiFi: ${e.message}", e)
            callback.onError("Lỗi tắt WiFi: ${e.message}")
        }
    }

    /**
     * Chuyển đổi trạng thái WiFi
     */
    fun toggleWifi(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Toggling WiFi")
            
            if (wifiManager.isWifiEnabled) {
                disableWifi(callback)
            } else {
                enableWifi(callback)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error toggling WiFi: ${e.message}", e)
            callback.onError("Lỗi chuyển đổi WiFi: ${e.message}")
        }
    }

    // ========== VOLUME CONTROL ==========

    /**
     * Tăng âm lượng
     */
    fun increaseVolume(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Increasing volume")
            
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            
            if (currentVolume < maxVolume) {
                val newVolume = (currentVolume + 1).coerceAtMost(maxVolume)
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, 0)
                Log.d(TAG, "Volume increased from $currentVolume to $newVolume")
                callback.onSuccess()
            } else {
                Log.d(TAG, "Volume is already at maximum")
                callback.onSuccess()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error increasing volume: ${e.message}", e)
            callback.onError("Lỗi tăng âm lượng: ${e.message}")
        }
    }

    /**
     * Giảm âm lượng
     */
    fun decreaseVolume(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Decreasing volume")
            
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            
            if (currentVolume > 0) {
                val newVolume = (currentVolume - 1).coerceAtLeast(0)
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, 0)
                Log.d(TAG, "Volume decreased from $currentVolume to $newVolume")
                callback.onSuccess()
            } else {
                Log.d(TAG, "Volume is already at minimum")
                callback.onSuccess()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error decreasing volume: ${e.message}", e)
            callback.onError("Lỗi giảm âm lượng: ${e.message}")
        }
    }

    /**
     * Đặt âm lượng cụ thể (0-100%)
     */
    fun setVolume(percentage: Int, callback: DeviceCallback) {
        try {
            Log.d(TAG, "Setting volume to $percentage%")
            
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            val targetVolume = ((percentage / 100.0) * maxVolume).toInt().coerceIn(0, maxVolume)
            
            audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, targetVolume, 0)
            Log.d(TAG, "Volume set to $targetVolume (${percentage}%)")
            callback.onSuccess()
        } catch (e: Exception) {
            Log.e(TAG, "Error setting volume: ${e.message}", e)
            callback.onError("Lỗi đặt âm lượng: ${e.message}")
        }
    }

    // ========== BRIGHTNESS CONTROL ==========

    /**
     * Tăng độ sáng màn hình
     */
    fun increaseBrightness(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Increasing brightness")
            
            val currentBrightness = Settings.System.getInt(
                context.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                0
            )
            
            val maxBrightness = 255
            val newBrightness = (currentBrightness + 25).coerceAtMost(maxBrightness)
            
            Settings.System.putInt(
                context.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                newBrightness
            )
            
            Log.d(TAG, "Brightness increased from $currentBrightness to $newBrightness")
            callback.onSuccess()
        } catch (e: Exception) {
            Log.e(TAG, "Error increasing brightness: ${e.message}", e)
            callback.onError("Lỗi tăng độ sáng: ${e.message}")
        }
    }

    /**
     * Giảm độ sáng màn hình
     */
    fun decreaseBrightness(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Decreasing brightness")
            
            val currentBrightness = Settings.System.getInt(
                context.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                0
            )
            
            val newBrightness = (currentBrightness - 25).coerceAtLeast(0)
            
            Settings.System.putInt(
                context.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                newBrightness
            )
            
            Log.d(TAG, "Brightness decreased from $currentBrightness to $newBrightness")
            callback.onSuccess()
        } catch (e: Exception) {
            Log.e(TAG, "Error decreasing brightness: ${e.message}", e)
            callback.onError("Lỗi giảm độ sáng: ${e.message}")
        }
    }

    /**
     * Đặt độ sáng cụ thể (0-100%)
     */
    fun setBrightness(percentage: Int, callback: DeviceCallback) {
        try {
            Log.d(TAG, "Setting brightness to $percentage%")
            
            val targetBrightness = ((percentage / 100.0) * 255).toInt().coerceIn(0, 255)
            
            Settings.System.putInt(
                context.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                targetBrightness
            )
            
            Log.d(TAG, "Brightness set to $targetBrightness (${percentage}%)")
            callback.onSuccess()
        } catch (e: Exception) {
            Log.e(TAG, "Error setting brightness: ${e.message}", e)
            callback.onError("Lỗi đặt độ sáng: ${e.message}")
        }
    }

    // ========== FLASH CONTROL ==========

    private var isFlashOn = false
    private var cameraManager: CameraManager? = null
    private var cameraId: String? = null

    init {
        initializeFlash()
    }

    private fun initializeFlash() {
        try {
            cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            // Tìm camera có flash
            cameraId = cameraManager?.cameraIdList?.find { id ->
                val characteristics = cameraManager?.getCameraCharacteristics(id)
                characteristics?.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            }
            Log.d(TAG, "Flash camera ID: $cameraId")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing flash: ${e.message}", e)
        }
    }

    /**
     * Bật đèn flash
     */
    fun enableFlash(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Enabling flash")
            
            if (cameraId != null) {
                cameraManager?.setTorchMode(cameraId!!, true)
                isFlashOn = true
                Log.d(TAG, "Flash enabled successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No camera with flash available")
                callback.onError("Thiết bị không có đèn flash")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error enabling flash: ${e.message}", e)
            callback.onError("Lỗi bật đèn flash: ${e.message}")
        }
    }

    /**
     * Tắt đèn flash
     */
    fun disableFlash(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Disabling flash")
            
            if (cameraId != null) {
                cameraManager?.setTorchMode(cameraId!!, false)
                isFlashOn = false
                Log.d(TAG, "Flash disabled successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No camera with flash available")
                callback.onError("Thiết bị không có đèn flash")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error disabling flash: ${e.message}", e)
            callback.onError("Lỗi tắt đèn flash: ${e.message}")
        }
    }

    /**
     * Chuyển đổi trạng thái đèn flash
     */
    fun toggleFlash(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Toggling flash")
            
            if (isFlashOn) {
                disableFlash(callback)
            } else {
                enableFlash(callback)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error toggling flash: ${e.message}", e)
            callback.onError("Lỗi chuyển đổi đèn flash: ${e.message}")
        }
    }

    // ========== UTILITY METHODS ==========

    /**
     * Lấy trạng thái WiFi
     */
    fun isWifiEnabled(): Boolean {
        return wifiManager.isWifiEnabled
    }

    /**
     * Lấy âm lượng hiện tại (0-100%)
     */
    fun getCurrentVolumePercentage(): Int {
        val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
        val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
        return ((currentVolume.toFloat() / maxVolume) * 100).toInt()
    }

    /**
     * Lấy độ sáng hiện tại (0-100%)
     */
    fun getCurrentBrightnessPercentage(): Int {
        val currentBrightness = Settings.System.getInt(
            context.contentResolver,
            Settings.System.SCREEN_BRIGHTNESS,
            0
        )
        return ((currentBrightness.toFloat() / 255) * 100).toInt()
    }

    /**
     * Lấy trạng thái đèn flash
     */
    fun isFlashEnabled(): Boolean {
        return isFlashOn
    }

    /**
     * Reset tất cả về mặc định
     */
    fun resetToDefaults(callback: DeviceCallback) {
        try {
            Log.d(TAG, "Resetting device to defaults")
            
            // Reset volume to 50%
            setVolume(50, object : DeviceCallback {
                override fun onSuccess() {
                    // Reset brightness to 50%
                    setBrightness(50, object : DeviceCallback {
                        override fun onSuccess() {
                            // Disable flash
                            disableFlash(object : DeviceCallback {
                                override fun onSuccess() {
                                    Log.d(TAG, "Device reset to defaults successfully")
                                    callback.onSuccess()
                                }
                                override fun onError(error: String) {
                                    callback.onError(error)
                                }
                            })
                        }
                        override fun onError(error: String) {
                            callback.onError(error)
                        }
                    })
                }
                override fun onError(error: String) {
                    callback.onError(error)
                }
            })
        } catch (e: Exception) {
            Log.e(TAG, "Error resetting to defaults: ${e.message}", e)
            callback.onError("Lỗi reset thiết bị: ${e.message}")
        }
    }
}
