#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DevelopTimeMethod {
    DevelopTime,
    DevelopTimeWarp,
    Default,
}