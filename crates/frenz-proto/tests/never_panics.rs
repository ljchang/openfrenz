//! No input on any characteristic may panic a decoder.
use frenz_proto::{Characteristic, Packet};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20_000))]
    #[test]
    fn arbitrary_bytes_never_panic(index in 0usize..Characteristic::ALL.len(), bytes in proptest::collection::vec(any::<u8>(), 0..300)) {
        let c = Characteristic::ALL[index];
        if let Ok(Packet::Response(r)) = Packet::decode(c, &bytes) {
            let _ = r.body();
        } else if let Ok(Packet::Status(s)) = Packet::decode(c, &bytes) {
            let _ = s.kind();
        }
    }

    #[test]
    fn exact_length_payloads_decode(index in 0usize..4, bytes in proptest::collection::vec(any::<u8>(), 158)) {
        // Every 158-byte motion payload is valid; every 5-byte battery/contact payload is valid.
        let c = [Characteristic::Accelerometer, Characteristic::Gyroscope, Characteristic::Battery, Characteristic::Contact][index];
        let len = if index < 2 { 158 } else { 5 };
        prop_assert!(Packet::decode(c, &bytes[..len]).is_ok());
    }
}
