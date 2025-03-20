#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SENSOR_PIN D3    // Chân kết nối cảm biến
#define BUZZER_PIN D0
#define RST_PIN D4     // Configurable, see typical pin layout above
#define SS_PIN D8    // Configurable, see typical pin layout above
#define SCREEN_WIDTH 128 // Độ rộng màn hình OLED
#define SCREEN_HEIGHT 64 // Độ cao màn hình OLED
#define OLED_RESET    -1 // Không sử dụng chân reset của OLED

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MFRC522 rfid(SS_PIN, RST_PIN); // Instance of the class
MFRC522::MIFARE_Key key;
String tag;
int validCount = 0; // bộ đếm hợp lệ 
int real = 0; // hiển thị hợp lệ trên web
int fake = 0; // hiển thị bất thường trên web 

ESP8266WebServer server(80); // Khởi tạo web server trên cổng 80

const char* ssid = "Thi 83 HTQ"; // Tên mạng Wi-Fi
const char* password = "Thi2k3er123456"; // Mật khẩu mạng Wi-Fi

void setup() {
  Serial.begin(9600);
  pinMode(SENSOR_PIN, INPUT);   // Cài đặt chân cảm biến là đầu vào
  SPI.begin(); // Init SPI bus
  rfid.PCD_Init(); // Init MFRC522
  pinMode(BUZZER_PIN, OUTPUT);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C); // Khởi động màn hình OLED
  delay(1000);

  // Kết nối vào mạng Wi-Fi
  WiFi.begin(ssid, password);
  Serial.print("Đang kết nối vào mạng Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("Kết nối thành công!");
  Serial.print("Địa chỉ IP: ");
  Serial.println(WiFi.localIP());

  // Thiết lập các đường dẫn HTTP
  server.on("/", handleRoot);
  server.begin();
  display.clearDisplay(); // Xóa màn hình OLED
  display.setTextSize(1); // Đặt kích thước chữ
  display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
  display.setCursor(0, 0); // Đặt vị trí con trỏ
  display.println(WiFi.localIP()); // Hiển thị "IP" trên màn hình OLED
  display.display(); // Hiển thị nội dung trên màn hình OLED
  delay(3000);
  display.clearDisplay(); // Xóa màn hình OLED
  display.setTextSize(1); // Đặt kích thước chữ
  display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
  display.setCursor(0, 0); // Đặt vị trí con trỏ
  display.println("Wait Card"); // Hiển thị "Wait Card" trên màn hình OLED
  display.display(); // Hiển thị nội dung trên màn hình OLED
}

void loop() {
  server.handleClient(); // Xử lý các yêu cầu từ trình duyệt web
  int sensorValue = digitalRead(SENSOR_PIN); // đọc giá trị cảm biến
  if (sensorValue == 0) {
    if (validCount == 0) {
      digitalWrite(BUZZER_PIN, HIGH);
      delay(1000);
      digitalWrite(BUZZER_PIN, LOW);
      fake++;
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("Warning !!!"); // Hiển thị "Warning !!!" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
      delay(4000);
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("Wait Card"); // Hiển thị "Wait Card" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
    } else {
      validCount--;
      Serial.println("Reset");
      real++;
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("OK"); // Hiển thị "OK" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
      delay(4000);
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("Wait Card"); // Hiển thị "Wait Card" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
    }
  }
  // đoạn này kiểm tra số thẻ, nếu thẻ đúng 2116224552 thì +1 validcount còn sai thì buzzer kêu 
  if (!rfid.PICC_IsNewCardPresent())
    return;
  if (rfid.PICC_ReadCardSerial()) {
    for (byte i = 0; i < 4; i++) {
      tag += rfid.uid.uidByte[i];
    }
    Serial.println(tag);
    if (tag == "2116224552") {
      Serial.println("Access Granted!");
      if (validCount < 1) {
        validCount++; // Tăng số lần hợp lệ
      }
      Serial.println(validCount);
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("Correct :)"); // Hiển thị "Correct" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
    } else {
      display.clearDisplay(); // Xóa màn hình OLED
      display.setTextSize(1); // Đặt kích thước chữ
      display.setTextColor(SSD1306_WHITE); // Đặt màu chữ
      display.setCursor(0, 0); // Đặt vị trí con trỏ
      display.println("In Correct :("); // Hiển thị "InCorrect" trên màn hình OLED
      display.display(); // Hiển thị nội dung trên màn hình OLED
      Serial.println("Access Denied!");
      Serial.println(validCount);
      digitalWrite(BUZZER_PIN, HIGH);
      delay(1000);
      digitalWrite(BUZZER_PIN, LOW);
    }

    delay(100);
    tag = "";
    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
  }
}

void handleRoot() {
  // Tạo nội dung HTML trả về trình duyệt
  String html = "<html><head><title>ESP8266 RFID</title>";
  html += "<meta charset='UTF-8'>";
  html += "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>";
  html += "</head><body>";
  html += "<h1>ESP8266 RFID</h1>";
  html += "<canvas id='myChart' width='100' height='100'></canvas>";
  html += "<script>";
  html += "var ctx = document.getElementById('myChart').getContext('2d');";
  html += "var myChart = new Chart(ctx, {type: 'pie', data: {labels: ['Hợp lệ', 'Bất thường'], datasets: [{data: [" + String(real) + ", " + String(fake) + "], backgroundColor: ['green', 'red']}]}});";
  html += "</script>";
  html += "</body></html>";

  // Gửi phản hồi HTTP với nội dung HTML
  server.send(200, "text/html", html);
}
