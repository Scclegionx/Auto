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
import org.json.JSONObject

class ContactAutomation(private val context: Context) {

    companion object {
        private const val TAG = "ContactAutomation"
    }

    /**
     * Entry Point: Nhận JSON từ CommandDispatcher và điều phối logic
     */
    suspend fun executeWithEntities(entities: JSONObject): String {
        Log.d(TAG, "Executing contact with entities: $entities")

        // Parse dữ liệu
        val action = entities.optString("ACTION", "").lowercase()
        val name = entities.optString("NAME", "")
        val phone = entities.optString("PHONE", "")
        val email = entities.optString("EMAIL", "")

        // Routing logic: Dựa vào action để gọi đúng hàm
        return when (action) {
            "insert", "add", "thêm", "them" -> {
                if (name.isEmpty()) {
                    throw Exception("Cần chỉ định tên liên hệ")
                }
                // Kiểm tra xem có dùng direct insert không
                val useDirect = entities.optBoolean("DIRECT", false)
                if (useDirect) {
                    insertContactDirect(name, if (phone.isEmpty()) null else phone, if (email.isEmpty()) null else email)
                } else {
                    insertContact(name, if (phone.isEmpty()) null else phone, if (email.isEmpty()) null else email)
                }
            }
            "pick", "chọn", "chon" -> {
                pickContact()
            }
            "open", "mở", "mo" -> {
                openContactsApp()
            }
            else -> {
                throw Exception("Không hỗ trợ thao tác với danh bạ: $action")
            }
        }
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
                "Đã mở màn hình thêm liên hệ: $name"
            } else {
                Log.e(TAG, "No app available to handle contact insert")
                throw Exception("Không tìm thấy ứng dụng để thêm danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact: ${e.message}", e)
            throw Exception("Lỗi thêm danh bạ: ${e.message}")
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
                "Đã mở danh sách liên hệ để chọn"
            } else {
                Log.e(TAG, "No app available to handle contact pick")
                throw Exception("Không tìm thấy ứng dụng để chọn danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error picking contact: ${e.message}", e)
            throw Exception("Lỗi chọn danh bạ: ${e.message}")
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
                "Đã mở ứng dụng danh bạ"
            } else {
                Log.e(TAG, "No contacts app available")
                throw Exception("Không tìm thấy ứng dụng danh bạ")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error opening contacts app: ${e.message}", e)
            throw Exception("Lỗi mở ứng dụng danh bạ: ${e.message}")
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
                throw Exception("Cần cấp quyền WRITE_CONTACTS để thêm liên hệ tự động")
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
            "Đã thêm liên hệ $name vào danh bạ"
        } catch (e: Exception) {
            Log.e(TAG, "Error inserting contact directly: ${e.message}", e)
            throw Exception("Lỗi thêm liên hệ: ${e.message}")
        }
    }
}
