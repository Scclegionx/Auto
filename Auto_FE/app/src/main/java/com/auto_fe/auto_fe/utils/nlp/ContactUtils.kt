package com.auto_fe.auto_fe.utils.nlp

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.provider.ContactsContract
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * Utility functions cho Contact operations
 * Các hàm helper để tìm kiếm và xử lý danh bạ
 */
object ContactUtils {
    
    /**
     * Kiểm tra xem chuỗi có phải là số điện thoại không
     */
    fun isPhoneNumber(input: String): Boolean {
        return input.matches(Regex("^[+]?[0-9\\s\\-\\(\\)]+$"))
    }
    
    /**
     * Tìm số điện thoại từ tên liên hệ (fuzzy search)
     */
    fun findPhoneNumberByName(context: Context, contactName: String): String {
        Log.d("ContactUtils", "Searching for contact: $contactName")

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("ContactUtils", "READ_CONTACTS permission not granted")
            return ""
        }

        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )

        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
        val selectionArgs = arrayOf("%$contactName%")

        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )

        cursor?.use {
            if (it.moveToFirst()) {
                val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                return phoneNumber
            }
        }

        Log.d("ContactUtils", "No contact found for: $contactName")
        return ""
    }
    
    /**
     * Tìm contact chính xác 100% và trả về phone number
     * Dùng cho State Machine khi đã có exact match
     * Tìm cả case-insensitive và trim khoảng trắng
     */
    fun findExactContactWithPhone(context: Context, contactName: String): Pair<String, String>? {
        Log.d("ContactUtils", "Searching for exact contact: $contactName")

        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("ContactUtils", "READ_CONTACTS permission not granted")
            return null
        }

        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.NUMBER,
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
        )

        // Tìm tất cả contacts và so sánh case-insensitive
        val cursor: Cursor? = context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            null,
            null,
            null
        )

        val normalizedSearchName = contactName.trim().lowercase()
        
        cursor?.use {
            while (it.moveToNext()) {
                val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                val normalizedDisplayName = displayName.trim().lowercase()
                
                // So sánh exact match (case-insensitive, trim)
                if (normalizedDisplayName == normalizedSearchName) {
                    val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                    Log.d("ContactUtils", "Found exact contact: $displayName with phone: $phoneNumber")
                    return Pair(displayName, phoneNumber)
                }
            }
        }

        Log.d("ContactUtils", "No exact contact found for: $contactName")
        return null
    }
    
    /**
     * Danh sách từ cần bỏ khi tách tên
     */
    private val excludedWords = setOf(
        "cháu", "anh", "chị", "em", "bác", "cô", "dì", "thầy", "cô giáo",
        "bà", "ông", "nội", "ngoại", "cậu", "mợ", "dượng", "ba",
        "con", "chú", "bác", "cô", "dì", "thầy", "cô giáo", "bạn", "bạn bè"
    )

    /**
     * Tách từ thông minh - bỏ các từ không phải tên riêng
     */
    fun smartWordParsing(fullName: String): List<String> {
        val words = fullName.trim().split("\\s+".toRegex())
        return words.filter { word ->
            val cleanWord = word.lowercase().trim()
            cleanWord.isNotEmpty() && !excludedWords.contains(cleanWord)
        }
    }

    /**
     * Tìm kiếm danh bạ fuzzy - tìm tất cả danh bạ chứa từ khóa
     */
    fun findSimilarContacts(context: Context, searchWords: List<String>): List<String> {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            return emptyList()
        }

        val similarContacts = mutableSetOf<String>() // Dùng Set để tránh duplicate

        for (word in searchWords) {
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
            )

            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$word%")

            val cursor: Cursor? = context.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                null
            )

            cursor?.use {
                while (it.moveToNext()) {
                    val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    similarContacts.add(displayName)
                }
            }
        }

        return similarContacts.toList()
    }
    
    /**
     * Data class để đại diện cho một liên hệ tương tự
     */
    data class SimilarContact(
        val name: String,
        val phoneNumber: String
    )
    
    /**
     * Tìm kiếm fuzzy các liên hệ tương tự dựa trên input
     * Tự động tách từ khóa (bỏ các từ như "cháu", "anh", v.v.) và tìm các liên hệ chứa từ khóa
     * @param context Context
     * @param input Tên liên hệ đầu vào (ví dụ: "cháu vương")
     * @return Danh sách các liên hệ tương tự với số điện thoại, hoặc emptyList nếu không tìm thấy
     */
    fun findSimilarContactsWithPhone(context: Context, input: String): List<SimilarContact> {
        Log.d("ContactUtils", "Finding similar contacts for input: $input")
        
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS)
            != PackageManager.PERMISSION_GRANTED) {
            Log.e("ContactUtils", "READ_CONTACTS permission not granted")
            return emptyList()
        }
        
        // Tách từ khóa từ input (bỏ các từ như "cháu", "anh", v.v.)
        val searchWords = smartWordParsing(input)
        
        if (searchWords.isEmpty()) {
            Log.d("ContactUtils", "No search words after parsing")
            return emptyList()
        }
        
        Log.d("ContactUtils", "Search words: $searchWords")
        
        // Tìm tất cả liên hệ chứa các từ khóa này
        val similarContacts = mutableMapOf<String, String>() // Map<name, phoneNumber> để tránh duplicate
        
        for (word in searchWords) {
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.NUMBER,
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME
            )
            
            val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?"
            val selectionArgs = arrayOf("%$word%")
            
            val cursor: Cursor? = context.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                selection,
                selectionArgs,
                null
            )
            
            cursor?.use {
                while (it.moveToNext()) {
                    val displayName = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME))
                    val phoneNumber = it.getString(it.getColumnIndexOrThrow(ContactsContract.CommonDataKinds.Phone.NUMBER))
                    // Dùng Map để đảm bảo mỗi tên chỉ xuất hiện một lần (lấy số điện thoại đầu tiên tìm thấy)
                    if (!similarContacts.containsKey(displayName)) {
                        similarContacts[displayName] = phoneNumber
                    }
                }
            }
        }
        
        val result = similarContacts.map { (name, phone) ->
            SimilarContact(name, phone)
        }
        
        Log.d("ContactUtils", "Found ${result.size} similar contacts: ${result.map { it.name }}")
        return result
    }
    
    /**
     * Tìm kiếm thông minh: Thử tìm exact match trước, nếu không có thì tìm fuzzy
     * @param context Context
     * @param contactName Tên liên hệ cần tìm
     * @return Pair<name, phoneNumber> nếu tìm thấy exact match, null nếu không
     */
    fun smartFindContact(context: Context, contactName: String): Pair<String, String>? {
        // Thử tìm exact match trước
        val exactMatch = findExactContactWithPhone(context, contactName)
        if (exactMatch != null) {
            return exactMatch
        }
        
        // Nếu không có exact match, trả về null để caller có thể gọi findSimilarContactsWithPhone
        return null
    }
}

