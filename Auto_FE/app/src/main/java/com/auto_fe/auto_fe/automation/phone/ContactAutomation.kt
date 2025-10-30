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

class ContactAutomation(private val context: Context) {

    interface ContactCallback {
        fun onSuccess()
        fun onError(error: String)
    }

    companion object {
        private const val TAG = "ContactAutomation"
    }

    /**
     * Chỉnh sửa liên hệ hiện có
     * @param contactUri URI của liên hệ cần chỉnh sửa
     * @param name Tên mới (optional)
     * @param phone Số điện thoại mới (optional)
     * @param email Email mới (optional)
     * @param callback Callback để nhận kết quả
     */
    fun editContact(
        contactUri: Uri,
        name: String? = null,
        phone: String? = null,
        email: String? = null,
        callback: ContactCallback
    ) {
        try {
            Log.d(TAG, "editContact called with URI: $contactUri")
            Log.d(TAG, "Edit data - name: $name, phone: $phone, email: $email")

            val intent = Intent(Intent.ACTION_EDIT).apply {
                data = contactUri
                
                // Thêm thông tin cần chỉnh sửa vào extras
                name?.let { putExtra(ContactsContract.Intents.Insert.NAME, it) }
                phone?.let { putExtra(ContactsContract.Intents.Insert.PHONE, it) }
                email?.let { putExtra(ContactsContract.Intents.Insert.EMAIL, it) }
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contact edit intent started successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No app available to handle contact edit")
                callback.onError("Không tìm thấy ứng dụng để chỉnh sửa danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error editing contact: ${e.message}", e)
            callback.onError("Lỗi chỉnh sửa danh bạ: ${e.message}")
        }
    }

    /**
     * Thêm liên hệ mới
     * @param name Tên liên hệ
     * @param phone Số điện thoại (optional)
     * @param email Email (optional)
     * @param callback Callback để nhận kết quả
     */
    fun insertContact(
        name: String,
        phone: String? = null,
        email: String? = null,
        callback: ContactCallback
    ) {
        try {
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
                callback.onSuccess()
            } else {
                Log.e(TAG, "No app available to handle contact insert")
                callback.onError("Không tìm thấy ứng dụng để thêm danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact: ${e.message}", e)
            callback.onError("Lỗi thêm danh bạ: ${e.message}")
        }
    }

    /**
     * Mở danh sách liên hệ để chọn (ACTION_PICK)
     * @param callback Callback để nhận kết quả
     */
    fun pickContact(callback: ContactCallback) {
        try {
            Log.d(TAG, "pickContact called")

            val intent = Intent(Intent.ACTION_PICK).apply {
                type = ContactsContract.Contacts.CONTENT_TYPE
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contact pick intent started successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No app available to handle contact pick")
                callback.onError("Không tìm thấy ứng dụng để chọn danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error picking contact: ${e.message}", e)
            callback.onError("Lỗi chọn danh bạ: ${e.message}")
        }
    }

    /**
     * Mở ứng dụng danh bạ mặc định
     * @param callback Callback để nhận kết quả
     */
    fun openContactsApp(callback: ContactCallback) {
        try {
            Log.d(TAG, "openContactsApp called")

            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = ContactsContract.Contacts.CONTENT_URI
            }

            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                Log.d(TAG, "Contacts app opened successfully")
                callback.onSuccess()
            } else {
                Log.e(TAG, "No contacts app available")
                callback.onError("Không tìm thấy ứng dụng danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error opening contacts app: ${e.message}", e)
            callback.onError("Lỗi mở ứng dụng danh bạ: ${e.message}")
        }
    }

    /**
     * Kiểm tra xem có ứng dụng nào có thể xử lý danh bạ không
     */
    fun isContactsAppAvailable(): Boolean {
        return try {
            val intent = Intent(Intent.ACTION_INSERT).apply {
                type = ContactsContract.Contacts.CONTENT_TYPE
            }
            intent.resolveActivity(context.packageManager) != null
        } catch (e: Exception) {
            Log.e(TAG, "Error checking contacts app availability: ${e.message}")
            false
        }
    }

    /**
     * Lấy thông tin chi tiết về khả năng quản lý danh bạ
     */
    fun getContactsInfo(): Map<String, Any> {
        return mapOf(
            "contacts_app_available" to isContactsAppAvailable(),
            "can_insert_contact" to isContactsAppAvailable(),
            "can_edit_contact" to isContactsAppAvailable(),
            "can_pick_contact" to isContactsAppAvailable()
        )
    }

    /**
     * Thêm liên hệ trực tiếp không cần UI (yêu cầu quyền WRITE_CONTACTS)
     */
    fun insertContactDirect(name: String, phone: String?, email: String?, callback: ContactCallback) {
        try {
            Log.d(TAG, "insertContactDirect called with name: $name, phone: $phone, email: $email")

            if (ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_CONTACTS)
                != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "WRITE_CONTACTS permission not granted")
                callback.onError("Cần cấp quyền WRITE_CONTACTS để thêm liên hệ tự động")
                return
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
            callback.onSuccess()
        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact directly: ${e.message}", e)
            callback.onError("Lỗi thêm liên hệ: ${e.message}")
        }
    }
}
