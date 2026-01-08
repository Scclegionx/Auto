package com.auto_fe.auto_fe.automation.phone

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.ContactsContract
import android.util.Log
import android.Manifest
import android.content.ContentProviderOperation
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat
import com.auto_fe.auto_fe.base.ConfirmationRequirement
import org.json.JSONObject

class ContactAutomation(private val context: Context) {

    companion object {
        private const val TAG = "ContactAutomation"
    }

    /**
     * Bắt đầu luồng thêm liên hệ tự động
     * Hỏi tên liên hệ, sau đó hỏi số điện thoại, rồi mở màn hình thêm liên hệ
     */
    suspend fun startAddContactFlow(): String {
        Log.d(TAG, "Starting add contact flow")
        
        // Bắt đầu luồng thêm liên hệ tự động - hỏi tên trước
        throw ConfirmationRequirement(
            originalInput = "",
            confirmationQuestion = "Dạ, hãy nói tên liên hệ ạ.",
            onConfirmed = {
                throw Exception("Contact name collection - should be handled in workflow")
            },
            isMultipleContacts = false,
            actionType = "contact_add_name",
            actionData = ""
        )
    }

    /**
     * Thêm liên hệ với dữ liệu đã có (public để gọi từ AutomationWorkflowManager)
     */
    fun insertContactWithData(name: String, phone: String): String {
        return insertContact(name, phone, null)
    }
    
    /**
     * Thêm liên hệ mới
     * @param name Tên liên hệ
     * @param phone Số điện thoại (optional)
     * @param email Email (optional)
     */
    private fun insertContact(
        name: String,
        phone: String? = null,
        email: String? = null
    ): String {
        return try {
            Log.d(TAG, "insertContact called with name: $name, phone: $phone, email: $email")

            val intent = Intent(Intent.ACTION_INSERT).apply {
                type = ContactsContract.Contacts.CONTENT_TYPE
                
                // Thêm thông tin liên hệ vào extras
                putExtra(ContactsContract.Intents.Insert.NAME, name)
                phone?.let { putExtra(ContactsContract.Intents.Insert.PHONE, it) }
                email?.let { putExtra(ContactsContract.Intents.Insert.EMAIL, it) }
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contact insert intent started successfully")
                "Dạ, đã mở màn hình thêm liên hệ $name ạ. Bác hãy kiểm tra lại thông tin và bấm nút lưu"
            } else {
                Log.e(TAG, "No app available to handle contact insert")
                throw Exception("Dạ, con không tìm thấy ứng dụng để thêm danh bạ ạ.")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact: ${e.message}", e)
            throw Exception("Dạ, con không thể mở màn hình thêm danh bạ ạ.")
        }
    }

    /**
     * Mở danh sách liên hệ để chọn (ACTION_PICK)
     */
    private fun pickContact(): String {
        return try {
            Log.d(TAG, "pickContact called")

            val intent = Intent(Intent.ACTION_PICK).apply {
                type = ContactsContract.Contacts.CONTENT_TYPE
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contact pick intent started successfully")
                "Dạ, đã mở danh sách liên hệ để chọn ạ."
            } else {
                Log.e(TAG, "No app available to handle contact pick")
                throw Exception("Dạ, con không tìm thấy ứng dụng để chọn danh bạ ạ.")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error picking contact: ${e.message}", e)
            throw Exception("Dạ, con không thể mở danh sách liên hệ ạ.")
        }
    }

    /**
     * Mở ứng dụng danh bạ mặc định
     */
    private fun openContactsApp(): String {
        return try {
            Log.d(TAG, "openContactsApp called")

            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = ContactsContract.Contacts.CONTENT_URI
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contacts app opened successfully")
                "Dạ, đã mở ứng dụng danh bạ ạ."
            } else {
                Log.e(TAG, "No contacts app available")
                throw Exception("Dạ, con không tìm thấy ứng dụng danh bạ ạ.")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error opening contacts app: ${e.message}", e)
            throw Exception("Dạ, con không thể mở ứng dụng danh bạ ạ.")
        }
    }

    /**
     * Thêm liên hệ trực tiếp không cần UI (yêu cầu quyền WRITE_CONTACTS)
     */
    private fun insertContactDirect(name: String, phone: String?, email: String?): String {
        return try {
            Log.d(TAG, "insertContactDirect called with name: $name, phone: $phone, email: $email")

            if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "WRITE_CONTACTS permission not granted")
                throw Exception("Dạ, con cần quyền thêm liên hệ để thực hiện lệnh này ạ.")
            }

            val operations = ArrayList<ContentProviderOperation>()

            // 1) Tạo RawContact
            operations.add(
                ContentProviderOperation.newInsert(ContactsContract.RawContacts.CONTENT_URI)
                    .withValue(ContactsContract.RawContacts.ACCOUNT_TYPE, null)
                    .withValue(ContactsContract.RawContacts.ACCOUNT_NAME, null)
                    .build()
            )

            // 2) Thêm tên
            operations.add(
                ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                    .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                    .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.StructuredName.CONTENT_ITEM_TYPE)
                    .withValue(ContactsContract.CommonDataKinds.StructuredName.DISPLAY_NAME, name)
                    .build()
            )

            // 3) Thêm số điện thoại nếu có
            if (!phone.isNullOrEmpty()) {
                operations.add(
                    ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                        .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                        .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Phone.CONTENT_ITEM_TYPE)
                        .withValue(ContactsContract.CommonDataKinds.Phone.NUMBER, phone)
                        .withValue(ContactsContract.CommonDataKinds.Phone.TYPE, ContactsContract.CommonDataKinds.Phone.TYPE_MOBILE)
                        .build()
                )
            }

            // 4) Thêm email nếu có
            if (!email.isNullOrEmpty()) {
                operations.add(
                    ContentProviderOperation.newInsert(ContactsContract.Data.CONTENT_URI)
                        .withValueBackReference(ContactsContract.Data.RAW_CONTACT_ID, 0)
                        .withValue(ContactsContract.Data.MIMETYPE, ContactsContract.CommonDataKinds.Email.CONTENT_ITEM_TYPE)
                        .withValue(ContactsContract.CommonDataKinds.Email.ADDRESS, email)
                        .withValue(ContactsContract.CommonDataKinds.Email.TYPE, ContactsContract.CommonDataKinds.Email.TYPE_WORK)
                        .build()
                )
            }

            // Apply batch
            context.contentResolver.applyBatch(ContactsContract.AUTHORITY, operations)
            Log.d(TAG, "Contact inserted successfully via ContentProvider")
            "Dạ, đã thêm liên hệ $name vào danh bạ ạ."
        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact directly: ${e.message}", e)
            throw Exception("Dạ, con không thể thêm liên hệ ạ.")
        }
    }
}
