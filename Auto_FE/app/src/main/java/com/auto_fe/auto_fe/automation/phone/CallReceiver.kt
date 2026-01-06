package com.auto_fe.auto_fe.automation.phone

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import android.util.Log
import androidx.core.content.ContextCompat

class CallReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != TelephonyManager.ACTION_PHONE_STATE_CHANGED) return

        val state = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
        val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER)

        Log.d("CallReceiver", "Phone state: $state, Number: $incomingNumber")

        val serviceIntent = Intent(context, CallAlertService::class.java).apply {
            action = when (state) {
                TelephonyManager.EXTRA_STATE_RINGING -> CallAlertService.ACTION_INCOMING_CALL
                TelephonyManager.EXTRA_STATE_OFFHOOK, 
                TelephonyManager.EXTRA_STATE_IDLE -> CallAlertService.ACTION_CALL_ENDED
                else -> ""
            }
            // Truyền số điện thoại sang Service để check contact bên đó
            if (!incomingNumber.isNullOrEmpty()) {
                putExtra(CallAlertService.EXTRA_PHONE_NUMBER, incomingNumber)
            }
        }

        // Dùng startForegroundService để đảm bảo service chạy được từ background (Android 8+)
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }
}