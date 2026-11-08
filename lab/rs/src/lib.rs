//! Define streaming contracts and the v1 PCM wire format; no neural model is included.

pub const RATE: u32 = 16_000;
pub const SIZE: usize = 320;
pub const HEAD: usize = 19;

/// Carry one audio frame on a session sequence.
pub struct Frame {
    pub seq: u64,
    pub samples: [f32; SIZE],
}

/// Carry generated audio, text, model scores, and measured processing time.
pub struct Output {
    pub audio: [f32; SIZE],
    pub user: String,
    pub agent: String,
    pub events: [f32; 5],
    pub elapsed_ms: f64,
}

/// Specify a stateful causal engine without implementing a neural backend.
pub trait Engine {
    /// Consume one current frame while preserving the session state.
    fn process(&mut self, frame: &Frame) -> Result<Output, &'static str>;
}

/// Encode finite PCM samples with the Python reference rounding convention.
pub fn pack(seq: u64, clock: u64, samples: &[f32]) -> Result<Vec<u8>, &'static str> {
    if samples.len() > u16::MAX as usize || samples.iter().any(|x| !x.is_finite()) {
        return Err("invalid samples");
    }
    let mut out = Vec::with_capacity(HEAD + samples.len() * 2);
    out.push(1);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&clock.to_le_bytes());
    out.extend_from_slice(&(samples.len() as u16).to_le_bytes());
    for x in samples {
        let pcm = (x.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16;
        out.extend_from_slice(&pcm.to_le_bytes());
    }
    Ok(out)
}

/// Represent a decoded wire sequence, sample clock, and PCM samples.
pub type Packet = (u64, u64, Vec<f32>);

/// Decode a validated packet without imposing the neural engine frame size.
pub fn unpack(packet: &[u8]) -> Result<Packet, &'static str> {
    if packet.len() < HEAD || packet[0] != 1 {
        return Err("invalid header");
    }
    let mut seq = [0_u8; 8];
    seq.copy_from_slice(&packet[1..9]);
    let mut clock = [0_u8; 8];
    clock.copy_from_slice(&packet[9..17]);
    let n = u16::from_le_bytes([packet[17], packet[18]]) as usize;
    if packet.len() != HEAD + n * 2 {
        return Err("invalid length");
    }
    let audio = packet[HEAD..].chunks_exact(2)
        .map(|x| i16::from_le_bytes([x[0], x[1]]) as f32 / 32767.0)
        .collect();
    Ok((u64::from_le_bytes(seq), u64::from_le_bytes(clock), audio))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Match fixed Python bytes including ties-to-even PCM rounding.
    #[test]
    fn golden_bytes() {
        let packet = pack(7, 960, &[-1.0, -0.5, 0.0, 0.5, 1.0]).unwrap();
        assert_eq!(packet, vec![1,7,0,0,0,0,0,0,0,192,3,0,0,0,0,0,0,5,0,1,128,0,192,0,0,0,64,255,127]);
        let (seq, clock, audio) = unpack(&packet).unwrap();
        assert_eq!((seq, clock), (7, 960));
        assert_eq!(audio.len(), 5);
        assert_eq!(audio[0], -1.0);
    }

    /// Reject truncation and nonfinite sample values.
    #[test]
    fn invalid_input() {
        assert!(unpack(&[1, 0]).is_err());
        assert!(pack(0, 0, &[f32::NAN]).is_err());
        let mut packet = pack(0, 0, &[0.0]).unwrap();
        packet.pop();
        assert!(unpack(&packet).is_err());
    }
}
