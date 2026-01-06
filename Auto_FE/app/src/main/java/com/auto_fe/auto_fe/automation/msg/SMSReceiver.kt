package com.auto_fe.auto_fe.automation.msg

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.provider.Telephony
import android.util.Log

class SMSReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Telephony.Sms.Intents.SMS_RECEIVED_ACTION) return

        Log.d("SMSReceiver", "SMS Received Broadcast detected")

        // Lấy nội dung tin nhắn từ Intent (Không cần query database)
        val messages = Telephony.Sms.Intents.getMessagesFromIntent(intent)
        if (messages.isNullOrEmpty()) return

        // Tin nhắn dài có thể bị chia thành nhiều phần, ta ghép lại
        val sender = messages[0].displayOriginatingAddress
        val body = StringBuilder()
        for (msg in messages) {
            body.append(msg.messageBody)
        }

        Log.d("SMSReceiver", "From: $sender, Body: $body")

        // Gửi sang Service để xử lý (Check danh bạ & Đọc)
        val serviceIntent = Intent(context, SmsAlertService::class.java).apply {
            action = SmsAlertService.ACTION_ALERT_SMS
            putExtra(SmsAlertService.EXTRA_PHONE_NUMBER, sender)
            putExtra(SmsAlertService.EXTRA_BODY, body.toString())
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }
}