mod application;

use crate::application::App;
use anyhow::Result;

fn main() -> Result<()> {
    App::run()
}
