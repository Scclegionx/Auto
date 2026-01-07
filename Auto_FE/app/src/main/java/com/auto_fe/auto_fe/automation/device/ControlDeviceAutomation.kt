package com.auto_fe.auto_fe.automation.device

import android.content.Context
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.util.Log
import com.auto_fe.auto_fe.utils.nlp.DeviceControlUtils
import org.json.JSONObject

class ControlDeviceAutomation(private val context: Context) {

    companion object {
        private const val TAG = "ControlDeviceAutomation"
    }

    private val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager

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
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     * Device: "đèn pin" hoặc "âm lượng"
     * Action: "bật"/"tắt" cho đèn và "tăng"/"giảm" cho âm lượng
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing device control with entities: $entities")

        // Parse dữ liệu
        val device = entities.optString("DEVICE", "")
        val action = entities.optString("ACTION", "")

        // Routing logic đơn giản: so sánh trực tiếp
        return when (device) {
            "đèn pin" -> {
                when (action) {
                    "bật" -> enableFlash()
                    "tắt" -> disableFlash()
                    else -> throw Exception("Không hiểu thao tác với đèn pin: $action")
                }
            }
            "âm lượng" -> {
                when (action) {
                    "tăng" -> increaseVolume()
                    "giảm" -> decreaseVolume()
                    else -> throw Exception("Không hiểu thao tác với âm lượng: $action")
                }
            }
            else -> {
                throw Exception("Không hỗ trợ điều khiển thiết bị: $device")
            }
        }
    }

    // ========== VOLUME CONTROL ==========

    /**
     * Tăng âm lượng
     */
    fun increaseVolume(): String {
        return try {
            Log.d(TAG, "Increasing volume")
            
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            val maxVolume = audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
            
            if (currentVolume < maxVolume) {
                val newVolume = (currentVolume + 1).coerceAtMost(maxVolume)
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, 0)
                val percentage = DeviceControlUtils.getCurrentVolumePercentage(context)
                Log.d(TAG, "Volume increased from $currentVolume to $newVolume")
                "Đã tăng âm lượng lên $percentage%"
            } else {
                Log.d(TAG, "Volume is already at maximum")
                "Âm lượng đã ở mức tối đa"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error increasing volume: ${e.message}", e)
            throw Exception("Lỗi tăng âm lượng: ${e.message}")
        }
    }

    /**
     * Giảm âm lượng
     */
    fun decreaseVolume(): String {
        return try {
            Log.d(TAG, "Decreasing volume")
            
            val currentVolume = audioManager.getStreamVolume(AudioManager.STREAM_MUSIC)
            
            if (currentVolume > 0) {
                val newVolume = (currentVolume - 1).coerceAtLeast(0)
                audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, newVolume, 0)
                val percentage = DeviceControlUtils.getCurrentVolumePercentage(context)
                Log.d(TAG, "Volume decreased from $currentVolume to $newVolume")
                "Đã giảm âm lượng xuống $percentage%"
            } else {
                Log.d(TAG, "Volume is already at minimum")
                "Âm lượng đã ở mức tối thiểu"
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error decreasing volume: ${e.message}", e)
            throw Exception("Lỗi giảm âm lượng: ${e.message}")
        }
    }

    // ========== FLASH CONTROL ==========

    /**
     * Bật đèn flash
     */
    fun enableFlash(): String {
        return try {
            Log.d(TAG, "Enabling flash")
            
            if (cameraId != null) {
                cameraManager?.setTorchMode(cameraId!!, true)
                isFlashOn = true
                Log.d(TAG, "Flash enabled successfully")
                "Đã bật đèn flash"
            } else {
                Log.e(TAG, "No camera with flash available")
                throw Exception("Thiết bị không có đèn flash")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error enabling flash: ${e.message}", e)
            throw Exception("Lỗi bật đèn flash: ${e.message}")
        }
    }

    /**
     * Tắt đèn flash
     */
    fun disableFlash(): String {
        return try {
            Log.d(TAG, "Disabling flash")
            
            if (cameraId != null) {
                cameraManager?.setTorchMode(cameraId!!, false)
                isFlashOn = false
                Log.d(TAG, "Flash disabled successfully")
                "Đã tắt đèn flash"
            } else {
                Log.e(TAG, "No camera with flash available")
                throw Exception("Thiết bị không có đèn flash")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error disabling flash: ${e.message}", e)
            throw Exception("Lỗi tắt đèn flash: ${e.message}")
        }
    }
}
