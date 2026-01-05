package com.example.Auto_BE.service;

import com.example.Auto_BE.dto.request.PrescriptionCreateRequest;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.cdimascio.dotenv.Dotenv;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Base64;

@Service
@Slf4j
@RequiredArgsConstructor
public class GeminiService {
    
    private static final Dotenv dotenv = Dotenv.configure()
            .directory("src/main/resources")
            .filename(".env")
            .ignoreIfMissing()
            .load();

    private final String apiKey = dotenv.get("GEMINI_API_KEY");
    private final ObjectMapper objectMapper;

    private static final String GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent";

    private static final String PROMPT = "Bạn là hệ thống nhận dạng và trích xuất dữ liệu y tế.\n" +
            "\n" +
            "Từ ảnh đơn thuốc (toa thuốc) được cung cấp, hãy đọc nội dung trong ảnh và trích xuất thông tin dưới dạng JSON theo cấu trúc sau:\n" +
            "\n" +
            "{\n" +
            "  \"name\": \"Tên đơn thuốc (nếu có, ví dụ: Đơn viêm da dày)\",\n" +
            "  \"description\": \"Ghi chú hoặc hướng dẫn tổng quát (ví dụ: Uống đều đặn trong 30 ngày, hoặc theo chỉ định bác sĩ)\",\n" +
            "  \"imageUrl\": null,\n" +
            "  \"medicationReminders\": [\n" +
            "    {\n" +
            "      \"name\": \"Tên thuốc để phân biệt không cần hiển thị hoạt chất (ví dụ: Atorvastatin)\",\n" +
            "      \"description\": \"Liều lượng và cách dùng (không rõ thì để trống, ví dụ: 1 viên sau ăn tối)\",\n" +
            "      \"type\": \"PRESCRIPTION\",\n" +
            "      \"reminderTimes\": [],\n" +
            "      \"daysOfWeek\": \"1111111\"\n" +
            "    }\n" +
            "  ]\n" +
            "}\n" +
            "\n" +
            "Quy tắc xử lý:\n" +
            "- Nếu đơn thuốc không ghi tên thì đặt name = \"Đơn thuốc không tiêu đề\".\n" +
            "- Nếu không có mô tả chung thì để description = null.\n" +
            "- Nếu tên thuốc hoặc mô tả có chứa thông tin tần suất hoặc thời điểm uống, hãy tự động gán giá trị cho \"reminderTimes\" theo quy tắc:\n" +
            "  - \"1 lần trong ngày\" = [\"08:00\"]\n" +
            "  - \"2 lần trong ngày\" = [\"08:00\",\"20:00\"]\n" +
            "  - \"3 lần trong ngày\" = [\"08:00\",\"12:00\",\"20:00\"]\n" +
            "  - \"4 lần trong ngày\" = [\"08:00\",\"12:00\",\"17:00\",\"21:00\"]\n" +
            "  - \"Buổi sáng\" = [\"08:00\"]\n" +
            "  - \"Buổi trưa\" = [\"12:00\"]\n" +
            "  - \"Buổi chiều\" = [\"17:00\"]\n" +
            "  - \"Buổi tối\" = [\"20:00\"]\n" +
            "  - \"Sáng và tối\" = [\"08:00\",\"20:00\"]\n" +
            "  - \"Sáng - trưa - tối\" = [\"08:00\",\"12:00\",\"20:00\"]\n" +
            "  - \"Sau ăn\" = tùy thời điểm, nếu không rõ = [\"12:00\"]\n" +
            "- Nếu không rõ thời điểm thì để mảng reminderTimes rỗng.\n" +
            "- Mặc định daysOfWeek = \"1111111\" (uống hàng ngày).\n" +
            "- Luôn đảm bảo output là JSON hợp lệ, không có mô tả thừa ngoài JSON.\n" +
            "- Không dịch, giữ nguyên ngôn ngữ tiếng Việt của thuốc và mô tả.\n" +
            "\n" +
            "Kết quả: chỉ xuất đối tượng JSON hợp lệ duy nhất.";

    public PrescriptionCreateRequest extractPrescriptionFromImage(MultipartFile imageFile) throws IOException, InterruptedException {
        // Convert image to base64
        String base64Image = Base64.getEncoder().encodeToString(imageFile.getBytes());
        String mimeType = imageFile.getContentType();

        // Build request body
        String requestBody = String.format("""
                {
                  "contents": [{
                    "parts": [
                      {"text": "%s"},
                      {
                        "inline_data": {
                          "mime_type": "%s",
                          "data": "%s"
                        }
                      }
                    ]
                  }]
                }
                """, PROMPT.replace("\n", "\\n").replace("\"", "\\\""), mimeType, base64Image);

        // Call Gemini API
        HttpClient client = HttpClient.newHttpClient();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(GEMINI_API_URL + "?key=" + apiKey))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();

        log.info("Calling Gemini API to extract prescription data...");
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            log.error("Gemini API error: {}", response.body());
            throw new RuntimeException("Failed to call Gemini API: " + response.body());
        }

        // Parse response
        String responseBody = response.body();
        log.info("Gemini API response: {}", responseBody);

        // Extract JSON from response
        String jsonContent = extractJsonFromGeminiResponse(responseBody);
        log.info("Extracted JSON: {}", jsonContent);

        // Parse to PrescriptionCreateRequest
        return objectMapper.readValue(jsonContent, PrescriptionCreateRequest.class);
    }

    private String extractJsonFromGeminiResponse(String responseBody) {
        try {
            // Parse Gemini response structure
            var jsonNode = objectMapper.readTree(responseBody);
            String text = jsonNode
                    .path("candidates")
                    .get(0)
                    .path("content")
                    .path("parts")
                    .get(0)
                    .path("text")
                    .asText();

            // Remove markdown code blocks if present
            text = text.replaceAll("```json\\s*", "").replaceAll("```\\s*", "").trim();

            return text;
        } catch (Exception e) {
            log.error("Failed to extract JSON from Gemini response", e);
            throw new RuntimeException("Failed to parse Gemini response", e);
        }
    }
}
