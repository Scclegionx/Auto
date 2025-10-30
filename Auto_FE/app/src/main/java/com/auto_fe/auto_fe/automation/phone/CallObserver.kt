package com.auto_fe.auto_fe.automation.phone

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.media.AudioManager
import android.provider.ContactsContract
import android.telephony.TelephonyManager
import android.util.Log
import com.auto_fe.auto_fe.audio.AudioRecorder

class CallObserver : BroadcastReceiver() {

    companion object {
        private const val TAG = "CallObserver"
        private var previousRingVolume: Int? = null
        private var loweredForUnknownCall: Boolean = false
    }

    override fun onReceive(context: Context, intent: Intent) {
        try {
            if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

            val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
            when (state) {
            TelephonyManager.EXTRA_STATE_RINGING -> {
                val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)
                Log.d(TAG, "Incoming call ringing. Number: ${incomingNumber ?: "unknown"}")

                val isUnknown = isUnknownNumber(context, incomingNumber)
                 if (isUnknown) {
                     Log.d(TAG, "Unknown number detected → lowering ring volume and starting alert service")
                     // Ngắt đọc SMS nếu đang chạy để ưu tiên cảnh báo cuộc gọi
                     try {
                         val stopSms = Intent(context, com.auto_fe.auto_fe.automation.msg.SmsAlertService::class.java).apply {
                             action = com.auto_fe.auto_fe.automation.msg.SmsAlertService.ACTION_STOP
                         }
                         if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                             context.startForegroundService(stopSms)
                         } else {
                             context.startService(stopSms)
                         }
                     } catch (_: Exception) {}

                     val prev = lowerRingVolumeTemporarily(context)
                     startAlertService(context, prev)
                } else {
                    Log.d(TAG, "Known contact → no action")
                }
            }
            TelephonyManager.EXTRA_STATE_OFFHOOK, TelephonyManager.EXTRA_STATE_IDLE -> {
                // Restore volume when call is answered or ended
                restoreRingVolumeIfNeeded(context)
            }
            }
        } catch (e: Exception) {
            Log.e(TAG, "onReceive error: ${e.message}", e)
        }
    }

    private fun isUnknownNumber(context: Context, number: String?): Boolean {
        if (number.isNullOrBlank()) {
            // Không có số → coi như không xác định, cảnh báo an toàn
            return true
        }
        return try {
            val uri = ContactsContract.PhoneLookup.CONTENT_FILTER_URI.buildUpon()
                .appendPath(number)
                .build()
            context.contentResolver.query(
                uri,
                arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME),
                null,
                null,
                null
            ).use { cursor ->
                cursor == null || !cursor.moveToFirst()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Contact lookup failed: ${e.message}")
            // Nếu không tra cứu được, fail-safe coi như số lạ
            true
        }
    }

    private fun lowerRingVolumeTemporarily(context: Context): Int? {
        try {
            val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
            if (audioManager.ringerMode != AudioManager.RINGER_MODE_NORMAL) {
                Log.d(TAG, "Ringer mode is not NORMAL; skip volume change")
                return null
            }

            val max = audioManager.getStreamMaxVolume(AudioManager.STREAM_RING)
            val current = audioManager.getStreamVolume(AudioManager.STREAM_RING)
            val target = (max * 0.5f).toInt().coerceAtLeast(1)

            if (previousRingVolume == null) previousRingVolume = current

            if (current > target) {
                audioManager.setStreamVolume(AudioManager.STREAM_RING, target, 0)
                loweredForUnknownCall = true
                Log.d(TAG, "Ring volume lowered from $current to $target")
            } else {
                Log.d(TAG, "Current volume ($current) <= target ($target); no change")
            }
            return previousRingVolume
        } catch (e: Exception) {
            Log.e(TAG, "Failed to adjust ring volume: ${e.message}", e)
            return null
        }
    }

    private fun restoreRingVolumeIfNeeded(context: Context) {
        if (!loweredForUnknownCall) return
        try {
            val restore = previousRingVolume
            if (restore != null) {
                val audioManager = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
                audioManager.setStreamVolume(AudioManager.STREAM_RING, restore, 0)
                Log.d(TAG, "Ring volume restored to $restore")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to restore ring volume: ${e.message}", e)
        } finally {
            previousRingVolume = null
            loweredForUnknownCall = false
        }
    }

    private fun startAlertService(context: Context, prevVolume: Int?) {
        try {
            val intent = Intent(context, CallAlertService::class.java).apply {
                action = CallAlertService.ACTION_ALERT_UNKNOWN
                prevVolume?.let { putExtra(CallAlertService.EXTRA_PREV_VOLUME, it) }
            }
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start CallAlertService: ${e.message}")
        }
    }
}


