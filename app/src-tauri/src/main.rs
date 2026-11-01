// 릴리스에서 콘솔 창을 띄우지 않습니다.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    zorogravity_app_lib::run()
}
