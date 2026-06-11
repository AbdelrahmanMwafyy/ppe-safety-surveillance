// ═══════════════════════════════════════════════════════════════════════
//  main.cpp — Camera Rail Controller (PlatformIO)
//  ESP32 38-pin | NEMA17 via A4988/DRV8825
//
//  WIRING:
//    STEP        → GPIO 12
//    DIR         → GPIO 14
//    ENABLE      → GPIO 13   (active LOW: LOW=enabled, HIGH=disabled)
//    Limit LEFT  → GPIO 27   (INPUT_PULLUP, LOW when pressed)
//    Limit RIGHT → GPIO 26   (INPUT_PULLUP, LOW when pressed)
//
//  PYTHON COMMANDS (received via TCP socket on port 8080):
//    'R' → move RIGHT (CW)
//    'L' → move LEFT  (CCW)
//    'S' → stop motor
//    'E' → emergency stop + disable motor (hold until 'R' or 'L' sent)
//
//  ESP32 REPLIES TO PYTHON:
//    'R' → right limit switch was hit  (Python triggers camera tilt)
//    'L' → left  limit switch was hit
//
//  SERIAL MONITOR COMMANDS (for testing without Python connected):
//    r → move right
//    l → move left
//    s → stop
//    e → emergency stop / disable motor
// ═══════════════════════════════════════════════════════════════════════
#include <Arduino.h>
#include <WiFi.h>

// ─── Wi-Fi Credentials ────────────────────────────────────────────────
const char* ssid     = "Hazem";
const char* password = "lotfy222";

// ─── Pins ─────────────────────────────────────────────────────────────
const int PIN_STEP    = 12;
const int PIN_DIR     = 14;
const int PIN_ENABLE  = 13;   // A4988 ENABLE — LOW = motor on, HIGH = motor off
const int PIN_LIM_L   = 27;   // Left  limit switch (INPUT_PULLUP)
const int PIN_LIM_R   = 26;   // Right limit switch (INPUT_PULLUP)

// ─── Motor Speed ──────────────────────────────────────────────────────
// Step delay in microseconds. Lower = faster.
// Start at 800, decrease if you need more speed (minimum ~300 for NEMA17).
const int STEP_DELAY_US = 600;

// ─── State ────────────────────────────────────────────────────────────
// 'R' = moving right, 'L' = moving left, 'S' = stopped, 'E' = emergency
char currentState = 'S';

WiFiServer server(8080);
WiFiClient client;

// ─── Helper: enable / disable motor driver ────────────────────────────
void enableMotor()  { digitalWrite(PIN_ENABLE, LOW);  }
void disableMotor() { digitalWrite(PIN_ENABLE, HIGH); }

// ─── Helper: send one character to Python (if connected) ──────────────
void replyToPython(char c) {
  if (client && client.connected()) {
    client.print(c);
    Serial.printf("[TX → Python] %c\n", c);
  }
}

// ─── Helper: apply a command from any source ──────────────────────────
void applyCommand(char cmd) {
  switch (cmd) {
    case 'R':
      Serial.println("[CMD] Move RIGHT");
      enableMotor();
      digitalWrite(PIN_DIR, HIGH);
      delayMicroseconds(20);   // DIR must settle before first STEP
      currentState = 'R';
      break;

    case 'L':
      Serial.println("[CMD] Move LEFT");
      enableMotor();
      digitalWrite(PIN_DIR, LOW);
      delayMicroseconds(20);
      currentState = 'L';
      break;

    case 'S':
      Serial.println("[CMD] Stop");
      currentState = 'S';
      // Motor remains enabled — driver holds position
      break;

    case 'E':
      Serial.println("[CMD] Emergency Stop — motor disabled");
      currentState = 'E';
      disableMotor();   // Cuts hold current — rail can be moved by hand
      break;
  }
}

// ─── Setup ────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(300);

  // Pin modes
  pinMode(PIN_STEP,   OUTPUT);
  pinMode(PIN_DIR,    OUTPUT);
  pinMode(PIN_ENABLE, OUTPUT);
  pinMode(PIN_LIM_L,  INPUT_PULLUP);
  pinMode(PIN_LIM_R,  INPUT_PULLUP);

  // Start with motor stopped and enabled
  digitalWrite(PIN_STEP, LOW);
  enableMotor();

  // Connect WiFi
  Serial.printf("\nConnecting to %s", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nWiFi OK — IP: %s\n", WiFi.localIP().toString().c_str());

  server.begin();
  Serial.println("TCP server started on port 8080");
  Serial.println("Waiting for Python connection...");
  Serial.println("Serial commands: r=right  l=left  s=stop  e=emergency");

  // ── Auto-start motor RIGHT on boot ──────────────────────────────────
  delay(5000);        // 5s settle time after WiFi connects
  applyCommand('R'); // motor starts immediately, Python takes over when connected
}

// ─── Main Loop ────────────────────────────────────────────────────────
void loop() {

  // ── 1. Accept new client if none connected ───────────────────────────
  if (!client || !client.connected()) {
    WiFiClient incoming = server.available();
    if (incoming) {
      client = incoming;
      client.setNoDelay(true);   // Disable Nagle — sends replies instantly
      Serial.println("[NET] Python connected.");
    }
  }

  // ── 2. Read command from Python ──────────────────────────────────────
  if (client && client.available()) {
    char cmd = (char)client.read();
    cmd = toupper(cmd);
    if (cmd == 'R' || cmd == 'L' || cmd == 'S' || cmd == 'E') {
      applyCommand(cmd);
    }
  }

  // ── 3. Read command from Serial Monitor (for standalone testing) ─────
  if (Serial.available()) {
    char cmd = (char)Serial.read();
    cmd = toupper(cmd);
    if (cmd == 'R' || cmd == 'L' || cmd == 'S' || cmd == 'E') {
      applyCommand(cmd);
    }
  }

  // ── 4. Emergency/stopped — do nothing ───────────────────────────────
  if (currentState == 'S' || currentState == 'E') {
    return;
  }

  // ── 5. Check limit switches BEFORE stepping ──────────────────────────
  bool leftHit  = (digitalRead(PIN_LIM_L) == LOW);
  bool rightHit = (digitalRead(PIN_LIM_R) == LOW);

  if (currentState == 'R' && rightHit) {
    currentState = 'S';                // Stop motor
    Serial.println("[LIMIT] RIGHT hit → stopping");
    replyToPython('R');               // Tell Python → triggers camera tilt
    return;
  }

  if (currentState == 'L' && leftHit) {
    currentState = 'S';
    Serial.println("[LIMIT] LEFT hit → stopping");
    replyToPython('L');
    return;
  }

  // ── 6. Step motor ────────────────────────────────────────────────────
  digitalWrite(PIN_STEP, HIGH);
  delayMicroseconds(STEP_DELAY_US);
  digitalWrite(PIN_STEP, LOW);
  delayMicroseconds(STEP_DELAY_US);
}