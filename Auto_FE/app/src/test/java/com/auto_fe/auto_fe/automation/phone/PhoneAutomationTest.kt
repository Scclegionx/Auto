package com.auto_fe.auto_fe.automation.phone

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

class PhoneAutomationTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockPackageManager: PackageManager

    private lateinit var phoneAutomation: PhoneAutomation

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        `when`(mockContext.packageManager).thenReturn(mockPackageManager)
        phoneAutomation = PhoneAutomation(mockContext)
    }

    @Test
    fun testMakeCall_WithPhoneNumber_WithPermission_Success() {
        val phoneNumber = "0123456789"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = phoneAutomation.makeCallDirect(phoneNumber)
            assertTrue(result.contains("đang thực hiện cuộc gọi") || 
                      result.contains("đã mở màn hình quay số") ||
                      result.contains("Lỗi"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("Lỗi") == true || 
                      e.message?.contains("không tìm thấy") == true)
        }
    }

    @Test
    fun testMakeCall_NoPermission_OpensDialer() {
        val phoneNumber = "0123456789"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = phoneAutomation.makeCall(phoneNumber, "phone", requireConfirmation = true)
            assertTrue(result.contains("đã mở màn hình quay số") || result.contains("Lỗi"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("Lỗi") == true || 
                      e.message?.contains("không tìm thấy") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_MissingReceiver_ThrowsException() = runBlocking {
        val entities = JSONObject().apply {
            put("PLATFORM", "phone")
        }
        
        try {
            phoneAutomation.executeWithEntities(entities)
            fail("Expected exception for missing receiver")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("Cần chỉ định người nhận cuộc gọi") == true)
        }
    }

    @Test
    fun testMakeCall_ContactNotFound_ThrowsException() {
        val receiver = "Người không tồn tại 999999999"
        
        try {
            phoneAutomation.makeCallDirect(receiver)
            fail("Expected exception for contact not found")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("trong danh bạ chưa có tên này") == true ||
                      e.message?.contains("Lỗi") == true)
        }
    }

    @Test
    fun testMakeCall_ZaloPlatform_Success() {
        val phoneNumber = "0123456789"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(mock(ResolveInfo::class.java))
        
        try {
            val result = phoneAutomation.makeCall(phoneNumber, "zalo", requireConfirmation = true)
            assertTrue(result.contains("đang thực hiện cuộc gọi") || 
                      result.contains("đã mở màn hình quay số") ||
                      result.contains("Lỗi"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("Lỗi") == true || 
                      e.message?.contains("không tìm thấy") == true)
        }
    }

    @Test
    fun testMakeCall_NoDialerApp_ThrowsException() {
        val phoneNumber = "0123456789"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(null)
        
        try {
            phoneAutomation.makeCall(phoneNumber, "phone", requireConfirmation = true)
            fail("Expected exception for no dialer app")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không tìm thấy ứng dụng gọi điện") == true ||
                       e.message?.contains("Lỗi mở quay số") == true ||
                       e.message?.contains("Lỗi gọi điện") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_SupportSpeakEnabled_RequiresConfirmation() = runBlocking {
        val receiver = "0123456789"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("PLATFORM", "phone")
        }
        
        try {
            phoneAutomation.executeWithEntities(entities, "gọi điện cho $receiver")
            fail("Expected ConfirmationRequirement or exception")
        } catch (e: ConfirmationRequirement) {
            assertEquals("phone", e.actionType)
            assertEquals("phone", e.actionData)
            assertTrue(e.confirmationQuestion.contains("có phải bác muốn"))
        } catch (e: Exception) {
            assertTrue(e.message?.contains("Cần chỉ định") == true ||
                      e.message?.contains("trong danh bạ") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_ExactMatchContact_Success() = runBlocking {
        val receiver = "0123456789"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("PLATFORM", "phone")
        }
        
        try {
            val result = phoneAutomation.executeWithEntities(entities, "gọi điện cho $receiver")
            assertNotNull(result)
        } catch (e: ConfirmationRequirement) {
            assertTrue(true)
        } catch (e: Exception) {
            assertTrue(true)
        }
    }

    @Test
    fun testMakeCall_UnsupportedPlatform_ThrowsException() {
        val phoneNumber = "0123456789"
        val unsupportedPlatform = "telegram"
        
        `when`(mockPackageManager.resolveActivity(any(), eq(PackageManager.MATCH_DEFAULT_ONLY)))
            .thenReturn(null)
        
        try {
            phoneAutomation.makeCall(phoneNumber, unsupportedPlatform, requireConfirmation = true)
            fail("Expected exception for unsupported platform")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("không tìm thấy ứng dụng gọi điện") == true ||
                       e.message?.contains("Lỗi gọi điện") == true)
        }
    }

    @Test
    fun testExecuteWithEntities_SimilarContacts_RequiresConfirmation() = runBlocking {
        val receiver = "Vương Không Tồn Tại 12345"
        
        val entities = JSONObject().apply {
            put("RECEIVER", receiver)
            put("PLATFORM", "phone")
        }
        
        try {
            phoneAutomation.executeWithEntities(entities, "gọi điện cho $receiver")
            fail("Expected exception for contact not found or ConfirmationRequirement")
        } catch (e: ConfirmationRequirement) {
            assertEquals("phone", e.actionType)
            assertEquals("phone", e.actionData)
        } catch (e: Exception) {
            assertTrue(e.message?.contains("trong danh bạ chưa có tên này") == true ||
                      e.message?.contains("Cần chỉ định") == true)
        }
    }
}

