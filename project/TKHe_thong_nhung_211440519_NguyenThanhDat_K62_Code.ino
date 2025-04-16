#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <time.h>

// Cau hinh chan
#define IR_SENSOR_PIN D3
#define BUZZER_PIN D0
#define RST_PIN D4
#define SS_PIN D8
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// Cau hinh cac linh kien
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
MFRC522 rfid(SS_PIN, RST_PIN);
ESP8266WebServer server(80);

// WiFi 
const char* ssid = "Datdatnguyen";
const char* password = "26092003";

// Cau hinh NTPClient (thoi gian thuc)
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 7 * 3600); // GMT+7 timezone

// UID (co the update them phan database)
const String validUIDs[] = {"2116224552", "2431272320"};
const String userNames[] = {"Nguyen Van A", "Le Thi B"};
const String departments[] = {"Phong ban ABC", "Phong ban XYZ"};
const int numValidUIDs = 2;

// Bien he thong
int totalCheckedIn = 0;
int violations = 0;
int earlyArrivals = 0;
int lateArrivals = 0;
bool personDetected = false;
unsigned long detectionTime = 0;
const int cardReadWaitTime = 3000; // 3 seconds wait time

// Dinh nghia ca lam viec de xac dinh som hay muon
#define MORNING_SHIFT_HOUR 7
#define AFTERNOON_SHIFT_HOUR 13

// Kiem tra trang thai quet
enum AttendanceStatus {
  VALID_ON_TIME,
  VALID_EARLY,
  VALID_LATE,
  INVALID
};

// He thong truy cap
struct AttendanceLog {
  String uid;
  String userName;
  String department;
  String timestamp;
  AttendanceStatus status;
  int shift; // 1 la sang, 2 la chieu
};

#define MAX_LOG 100
AttendanceLog attendanceLog[MAX_LOG];
int logCount = 0;

void setup() {
  Serial.begin(115200);
  
  // Dinh nghia chan
  pinMode(IR_SENSOR_PIN, INPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  
  // Dinh nghia, khoi tao SPI va RFID
  SPI.begin();
  rfid.PCD_Init();
  
  // Khoi tao OLED
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  
  // Ket noi WiFi
  WiFi.begin(ssid, password);
  display.setCursor(0, 0);
  display.println("Dang ket noi WiFi...");
  display.display();
  
  Serial.print("Dang ket noi WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi da ket noi - IP: " + WiFi.localIP().toString());
  
  // Khoi tao thoi gian
  timeClient.begin();
  configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov"); // GMT+7 timezone
  
  // Setup web
  server.on("/", handleRoot);
  server.on("/reset", handleReset);
  server.begin();
  
  // Hien thi tin nhan 
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("IP: " + WiFi.localIP().toString());
  display.println("He thong san sang");
  display.println("Cho quet the...");
  display.display();
}

void loop() {
  server.handleClient();
  timeClient.update();
  
  // Kiem tra xem co nguoi vao hay khong
  if (digitalRead(IR_SENSOR_PIN) == LOW && !personDetected) {
    personDetected = true;
    detectionTime = millis();
    
    // Canh bao co nguoi vao
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Phat hien nguoi!");
    display.println("Vui long quet the");
    display.println("trong vong 3 giay...");
    display.display();
    
    Serial.println("Phat hien nguoi. Dang cho quet the...");
  }
  
  // Neu co nguoi vao, kiem tra the
  if (personDetected) {
    // Neu co the
    if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
      String uid = "";
      for (byte i = 0; i < rfid.uid.size; i++) {
        uid += String(rfid.uid.uidByte[i], DEC);
      }
      Serial.println("Da phat hien the: " + uid);
      
      // Valid UID
      bool isValidUID = false;
      int userIndex = -1;
      
      for (int i = 0; i < numValidUIDs; i++) {
        if (validUIDs[i] == uid) {
          isValidUID = true;
          userIndex = i;
          break;
        }
      }
      
      // Kiem tra thoi gian
      time_t now;
      struct tm timeinfo;
      time(&now);
      localtime_r(&now, &timeinfo);
      int currentHour = timeinfo.tm_hour;
      
      // Kiem tra ca
      int currentShift = (currentHour < AFTERNOON_SHIFT_HOUR) ? 1 : 2;
      
      // Xac dinh trang thai co mat
      AttendanceStatus status = INVALID;
      
      if (isValidUID) {
        // neu la buoi sang
        if (currentShift == 1) {
          if (currentHour < MORNING_SHIFT_HOUR) {
            status = VALID_EARLY;
            earlyArrivals++;
          } else {
            status = VALID_LATE;
            lateArrivals++;
          }
        }
        // neu la buoi chieu
        else {
          if (currentHour < AFTERNOON_SHIFT_HOUR) {
            status = VALID_EARLY;
            earlyArrivals++;
          } else {
            status = VALID_LATE;
            lateArrivals++;
          }
        }
        
        totalCheckedIn++;
        String currentTime = getFormattedTime();
        
        // them vao log
        if (logCount < MAX_LOG) {
          attendanceLog[logCount].uid = uid;
          attendanceLog[logCount].userName = userNames[userIndex];
          attendanceLog[logCount].department = departments[userIndex];
          attendanceLog[logCount].timestamp = currentTime;
          attendanceLog[logCount].status = status;
          attendanceLog[logCount].shift = currentShift;
          logCount++;
        }
        
        // Hien thi truy cap 
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println("Truy cap duoc chap nhan");
        display.println("UID: " + uid);
        display.println(userNames[userIndex]);
        display.println(departments[userIndex]);
        display.print("Ca lam: ");
        display.println(currentShift == 1 ? "Sang" : "Chieu");
        
        // Hien thi thong tin
        display.print("Trang thai: ");
        if (status == VALID_EARLY) {
          display.println("Den som");
        } else if (status == VALID_LATE) {
          display.println("Den muon");
        }
        
        display.println(currentTime);
        display.display();
        
        Serial.println("The hop le: " + userNames[userIndex]);
      } else {
        // The khong hop le
        violations++;
        String currentTime = getFormattedTime();
        
        // Them vao log
        if (logCount < MAX_LOG) {
          attendanceLog[logCount].uid = uid;
          attendanceLog[logCount].userName = "Khong xac dinh";
          attendanceLog[logCount].department = "Khong xac dinh";
          attendanceLog[logCount].timestamp = currentTime;
          attendanceLog[logCount].status = INVALID;
          attendanceLog[logCount].shift = currentShift;
          logCount++;
        }
        
        // Hien thi khong hop le va bao coi keu
        display.clearDisplay();
        display.setCursor(0, 0);
        display.println("TU CHOI TRUY CAP");
        display.println("UID khong hop le: " + uid);
        display.println(currentTime);
        display.display();
        
        Serial.println("Phat hien the khong hop le!");
        buzzTwice();
      }
      
      // Reset
      personDetected = false;
      
      // RFID
      rfid.PICC_HaltA();
      rfid.PCD_StopCrypto1();
      
      // Show sau 3s
      delay(3000);
      showReadyScreen();
    }
    
    // Neu sau 3s khong co the
    if (millis() - detectionTime > cardReadWaitTime && personDetected) {
      // Ko co the quet
      violations++;
      String currentTime = getFormattedTime();
      
      // Lay thoi gian hien tai de xac dinh ca
      time_t now;
      struct tm timeinfo;
      time(&now);
      localtime_r(&now, &timeinfo);
      int currentHour = timeinfo.tm_hour;
      int currentShift = (currentHour < AFTERNOON_SHIFT_HOUR) ? 1 : 2;
      
      // Them vao log
      if (logCount < MAX_LOG) {
        attendanceLog[logCount].uid = "Khong co";
        attendanceLog[logCount].userName = "Khong quet the";
        attendanceLog[logCount].department = "Khong xac dinh";
        attendanceLog[logCount].timestamp = currentTime;
        attendanceLog[logCount].status = INVALID;
        attendanceLog[logCount].shift = currentShift;
        logCount++;
      }
      
      // Hien thi vi pham va bao coi keu
      display.clearDisplay();
      display.setCursor(0, 0);
      display.println("VI PHAM!");
      display.println("Khong quet the");
      display.println(currentTime);
      display.display();
      
      Serial.println("Khong quet the trong thoi gian quy dinh!");
      buzzTwice();
      
      // Reset 
      personDetected = false;
      
      // Hien thi 3s
      delay(3000);
      showReadyScreen();
    }
  }
}

void buzzTwice() {
  for (int i = 0; i < 2; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  }
}

void showReadyScreen() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("He thong san sang");
  display.println("Cho quet the...");
  display.println("IP: " + WiFi.localIP().toString());
  display.display();
}

String getFormattedTime() {
  time_t now;
  struct tm timeinfo;
  time(&now);
  localtime_r(&now, &timeinfo);
  
  char buffer[30];
  sprintf(buffer, "%02d/%02d/%04d %02d:%02d:%02d",
          timeinfo.tm_mday, timeinfo.tm_mon + 1, timeinfo.tm_year + 1900,
          timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
  return String(buffer);
}

String getStatusText(AttendanceStatus status) {
  switch (status) {
    case VALID_EARLY:
      return "Den som";
    case VALID_LATE:
      return "Den muon";
    case VALID_ON_TIME:
      return "Dung gio";
    case INVALID:
      return "Khong hop le";
    default:
      return "Khong xac dinh";
  }
}

String getStatusClass(AttendanceStatus status) {
  switch (status) {
    case VALID_EARLY:
      return "early";
    case VALID_LATE:
      return "late";
    case VALID_ON_TIME:
      return "ontime";
    case INVALID:
      return "invalid";
    default:
      return "";
  }
}

void handleRoot() {
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'><title>He thong cham cong</title>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'>";
  html += "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'>";
  html += "<style>";
  html += "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');";
  html += "* {box-sizing: border-box; margin: 0; padding: 0;}";
  html += "body {font-family: 'Roboto', sans-serif; background-color: #f5f5f5; color: #333; padding: 10px;}";
  html += ".container {width: 100%; max-width: 600px; margin: 0 auto; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}";
  html += "header {display: flex; flex-direction: column; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px;}";
  html += "h1 {color: #2c3e50; font-size: 1.5rem; margin-bottom: 8px;}";
  html += "h2 {color: #2c3e50; font-size: 1.3rem; margin: 15px 0 10px 0;}";
  html += ".time-info {font-size: 0.9rem; display: flex; align-items: center;}";
  html += ".time-info i {margin-right: 5px; color: #3498db;}";
  html += ".dashboard {display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px;}";
  html += ".stat-card {background: #fff; border-radius: 8px; padding: 15px 10px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}";
  html += ".stat-card i {font-size: 1.5rem; margin-bottom: 5px;}";
  html += ".stat-card h3 {font-size: 1.4rem; margin: 5px 0;}";
  html += ".stat-card p {color: #7f8c8d; margin: 0; font-size: 0.8rem;}";
  html += ".checkedin {background: linear-gradient(135deg, #43cea2, #185a9d); color: white;}";
  html += ".violations {background: linear-gradient(135deg, #ff4b1f, #ff9068); color: white;}";
  html += ".early {background: linear-gradient(135deg, #56ab2f, #a8e063); color: white;}";
  html += ".late {background: linear-gradient(135deg, #ffb347, #ffcc33); color: white;}";
  html += ".checkedin h3, .violations h3, .early h3, .late h3, .checkedin p, .violations p, .early p, .late p {color: white;}";
  html += ".table-container {overflow-x: auto; margin-bottom: 15px;}";
  html += "table {width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}";
  html += "th, td {padding: 10px 8px; text-align: left; border-bottom: 1px solid #ddd; font-size: 0.85rem;}";
  html += "th {background-color: #3498db; color: white; font-weight: 500;}";
  html += "tr:hover {background-color: #f5f5f5;}";
  html += ".early {color: #27ae60;}";
  html += ".late {color: #e67e22;}";
  html += ".ontime {color: #2980b9;}";
  html += ".invalid {color: #e74c3c;}";
  html += ".btn-group {display: flex; gap: 10px; margin-top: 15px;}";
  html += ".btn {flex: 1; padding: 10px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9rem; text-align: center; text-decoration: none;}";
  html += ".btn i {margin-right: 5px;}";
  html += ".btn:hover {background-color: #2980b9;}";
  html += ".btn-reset {background-color: #e74c3c;}";
  html += ".btn-reset:hover {background-color: #c0392b;}";
  html += "footer {margin-top: 20px; text-align: center; color: #7f8c8d; font-size: 0.8rem;}";
  html += "</style>";
  html += "<script>";
  html += "function refreshPage() {location.reload();}";
  html += "setInterval(refreshPage, 10000);"; // Auto refresh every 10 seconds
  html += "</script>";
  html += "</head><body>";
  
  html += "<div class='container'>";
  html += "<header>";
  html += "<h1><i class='fas fa-id-card'></i> He thong cham cong</h1>";
  html += "<div class='time-info'><i class='far fa-clock'></i> " + getFormattedTime() + "</div>";
  html += "</header>";
  
  html += "<div class='dashboard'>";
  html += "<div class='stat-card checkedin'><i class='fas fa-user-check'></i><h3>" + String(totalCheckedIn) + "</h3><p>Da cham cong</p></div>";
  html += "<div class='stat-card violations'><i class='fas fa-exclamation-triangle'></i><h3>" + String(violations) + "</h3><p>Vi pham</p></div>";
  html += "<div class='stat-card early'><i class='fas fa-hourglass-start'></i><h3>" + String(earlyArrivals) + "</h3><p>Den som</p></div>";
  html += "<div class='stat-card late'><i class='fas fa-hourglass-end'></i><h3>" + String(lateArrivals) + "</h3><p>Den muon</p></div>";
  html += "</div>";
  
  html += "<h2><i class='fas fa-list'></i> Nhat ky cham cong</h2>";
  html += "<div class='table-container'>";
  html += "<table>";
  html += "<tr>";
  html += "<th>Thoi gian</th>";
  html += "<th>Ten</th>";
  html += "<th>Ca</th>";
  html += "<th>Trang thai</th>";
  html += "</tr>";
  
  // Hien thi log 
  for (int i = logCount - 1; i >= 0; i--) {
    String statusClass = getStatusClass(attendanceLog[i].status);
    String statusText = getStatusText(attendanceLog[i].status);
    String shift = attendanceLog[i].shift == 1 ? "Sang" : "Chieu";
    
    html += "<tr>";
    html += "<td>" + attendanceLog[i].timestamp + "</td>";
    html += "<td>" + attendanceLog[i].userName + "</td>";
    html += "<td>" + shift + "</td>";
    html += "<td class='" + statusClass + "'>" + statusText + "</td>";
    html += "</tr>";
  }
  
  html += "</table>";
  html += "</div>";
  
  html += "<div class='btn-group'>";
  html += "<button class='btn' onclick='refreshPage()'><i class='fas fa-sync-alt'></i> Cap nhat</button>";
  html += "<a href='/reset' class='btn btn-reset'><i class='fas fa-trash'></i> Xoa du lieu</a>";
  html += "</div>";
  
  html += "<footer>";
  html += "He thong cham cong | IP: " + WiFi.localIP().toString();
  html += "</footer>";
  
  html += "</div>"; // Ket thuc Containers
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void handleReset() {
  logCount = 0;
  totalCheckedIn = 0;
  violations = 0;
  earlyArrivals = 0;
  lateArrivals = 0;
  
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta charset='UTF-8'><title>Dat lai he thong</title>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'>";
  html += "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'>";
  html += "<style>";
  html += "@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');";
  html += "body {font-family: 'Roboto', sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f5f5f5;}";
  html += ".reset-box {text-align: center; background: white; padding: 30px 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 90%; max-width: 400px;}";
  html += "h1 {color: #2c3e50; font-size: 1.5rem;}";
  html += "i.success {font-size: 4rem; color: #27ae60; margin-bottom: 15px;}";
  html += ".btn {display: inline-block; padding: 12px 24px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; text-decoration: none; transition: background 0.3s ease;}";
  html += ".btn:hover {background-color: #2980b9;}";
  html += ".btn i {margin-right: 5px;}";
  html += "</style>";
  html += "</head><body>";
  
  html += "<div class='reset-box'>";
  html += "<i class='fas fa-check-circle success'></i>";
  html += "<h1>Dat lai he thong thanh cong</h1>";
  html += "<p>Tat ca du lieu nhat ky va thong ke da duoc xoa.</p>";
  html += "<a href='/' class='btn'><i class='fas fa-home'></i> Quay lai</a>";
  html += "</div>";
  
  html += "</body></html>";
  
  server.send(200, "text/html", html);
  
  // Update len Oled
  showReadyScreen();
}