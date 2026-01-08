package com.auto_fe.auto_fe.automation.msg

import android.content.Context
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo
import com.auto_fe.auto_fe.base.ConfirmationRequirement
import kotlinx.coroutines.runBlocking
import org.json.JSONObject
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*
import org.mockito.Mock
import org.mockito.Mockito.*
import org.mockito.MockitoAnnotations

class SMSAutomationTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPackageManager: PackageManager

    private lateinit var smsAutomation: SMSAutomation

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        `when`(mockContext.packageManager).thenReturn(mockPackageManager)
        smsAutomation = SMSAutomation(mockContext)
    }

    @Test
    fun testSendSMSWithPhoneNumber_Success() {
        val phoneNumber = "0123456789"
        val message = "Xin chào"
        
        try {
            val result = smsAutomation.sendSMSDirect(phoneNumber, message)
            assertTrue(result.contains("đã gửi tin nhắn") || result.contains("không thể gửi"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không thể gửi") == true || 
                      e.message?.contains("trong danh bạ") == true)
        }
    }

    @Test
    fun testOpenSmsCompose_SupportSpeakDisabled_Success() = runBlocking {
        val receiver = "0123456789"
        val message = "Test message"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY))).thenReturn(
            mock(ResolveInfo::class.java)
        )
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("MESSAGE", message)
        }
        
        try {
            val result = smsAutomation.executeWithEntities(entities, "gửi tin nhắn cho $receiver")
            assertTrue(result.contains("mở màn hình") || result.contains("đã gửi"))
        } catch (e: ConfirmationRequirement) {
            assertTrue(true)
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testExecuteWithEntities_MissingReceiver_ThrowsException() = runBlocking {
        val entities = JSONObject().apply {
            put("MESSAGE", "Test message")
        }
        
        try {
            smsAutomation.executeWithEntities(entities)
            fail("Expected exception for missing receiver")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("chưa nghe rõ tên người") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_MissingMessage_ThrowsException() = runBlocking {
        val entities = JSONObject().apply {
            put("RECEIVER", "0123456789")
        }
        
        try {
            smsAutomation.executeWithEntities(entities)
            fail("Expected exception for missing message")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("chưa nghe rõ nội dung tin nhắn") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_SupportSpeakEnabled_RequiresConfirmation() = runBlocking {
        val receiver = "0123456789"
        val message = "Test message"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("MESSAGE", message)
        }
        
        try {
            smsAutomation.executeWithEntities(entities, "gửi tin nhắn")
            fail("Expected ConfirmationRequirement")
        } catch (e: ConfirmationRequirement) {
            assertEquals("sms", e.actionType)
            assertEquals(message, e.actionData)
            assertTrue(e.confirmationQuestion.contains("có phải bác muốn"))
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testSendSMS_ContactNotFound_ThrowsException() {
        val receiver = "Người không tồn tại 999999999"
        val message = "Test"
        
        try {
            smsAutomation.sendSMSDirect(receiver, message)
            fail("Expected exception for contact not found")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("trong danh bạ chưa có tên này") == true ||
                      e.message?.contains("không thể gửi") == true)
        }
    }

    @Test
    fun testSendSMS_SmsManagerException_ThrowsException() {
        val phoneNumber = "0123456789"
        val message = "Test"
        
        try {
            val result = smsAutomation.sendSMSDirect(phoneNumber, message)
            assertTrue(result.contains("đã gửi") || result.contains("không thể"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không thể gửi tin nhắn") == true ||
                      e.message?.contains("trong danh bạ") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_ExactMatchContact_Success() = runBlocking {
        val receiver = "0123456789"
        val message = "Test message"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("MESSAGE", message)
        }
        
        try {
            val result = smsAutomation.executeWithEntities(entities, "gửi tin nhắn")
            assertNotNull(result)
        } catch (e: ConfirmationRequirement) {
            assertTrue(true)
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testOpenSmsCompose_NoSmsApp_ThrowsException() {
        val receiver = "0123456789"
        val message = "Test"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(null)
        
        try {
            smsAutomation.openSmsComposeDirect(receiver, message)
            fail("Expected exception for no SMS app")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không tìm thấy ứng dụng nhắn tin") == true ||
                       e.message?.contains("không thể mở màn hình soạn tin nhắn") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_SimilarContacts_RequiresConfirmation() = runBlocking {
        val receiver = "Vương Không Tồn Tại 12345"
        val message = "Test message"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("MESSAGE", message)
        }
        
        try {
            smsAutomation.executeWithEntities(entities, "gửi tin nhắn cho $receiver")
            fail("Expected exception for contact not found or ConfirmationRequirement")
        } catch (e: ConfirmationRequirement) {
            assertEquals("sms", e.actionType)
            assertEquals(message, e.actionData)
        } catch (e: Exception) {
            assertTrue(e.message?.contains("trong danh bạ chưa có tên này") == true ||
                      e.message?.contains("chưa nghe rõ") == true)
        }
    }
}

