//! Reverse map for the per-NID unresolved-import stubs the linker plants.
//!
//! [`xps5x_firmware::UNRESOLVED_STUB_BASE`] + `i * 8` is the address written
//! into every relocation slot of the `i`th distinct unresolved NID. Unlike the
//! HLE trampolines, this region is deliberately **never mapped**: the guest
//! reaching it is not a call to service, it is the guest asking for something
//! nothing implements. The access violation is the signal — this module's only
//! job is to turn the faulting address back into the NID that names it, so the
//! runtime can say *which* import is missing instead of printing a bare
//! sentinel address.

use xps5x_firmware::{UNRESOLVED_STUB_BASE, UnresolvedStub};

/// Map a faulting address within the unresolved-stub region to the
/// [`UnresolvedStub`] it names: `idx = (fault_addr - UNRESOLVED_STUB_BASE) /
/// 8`, indexed into `stubs`.
///
/// Returns `None` if `fault_addr` precedes [`UNRESOLVED_STUB_BASE`], names an
/// index past `stubs.len()`, or is not 8-byte aligned relative to the base —
/// i.e. a genuine wild-pointer fault that merely happens to land in this
/// address range. Callers treat `None` as an ordinary guest fault.
pub(crate) fn resolve(fault_addr: u64, stubs: &[UnresolvedStub]) -> Option<&UnresolvedStub> {
    let offset = fault_addr.checked_sub(UNRESOLVED_STUB_BASE)?;
    // A slot address is exact: the linker never adds a relocation addend to a
    // stub (see `link_module`'s `Resolver::Unresolved` arm), precisely so this
    // inversion stays exact. A mid-slot address is therefore not one of ours.
    if !offset.is_multiple_of(8) {
        return None;
    }
    let idx = usize::try_from(offset / 8).ok()?;
    stubs.get(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stub(nid: u64, idx: u64) -> UnresolvedStub {
        UnresolvedStub {
            nid,
            library: Some("libc".to_string()),
            addr: UNRESOLVED_STUB_BASE + idx * 8,
        }
    }

    #[test]
    fn base_address_resolves_to_the_first_stub() {
        let stubs = vec![stub(0xAABB, 0), stub(0xCCDD, 1)];
        let got = resolve(UNRESOLVED_STUB_BASE, &stubs).expect("base names stub 0");
        assert_eq!(got.nid, 0xAABB);
    }

    #[test]
    fn nth_slot_resolves_to_the_nth_stub() {
        let stubs = vec![stub(0xAABB, 0), stub(0xCCDD, 1), stub(0xEEFF, 2)];
        let got = resolve(UNRESOLVED_STUB_BASE + 16, &stubs).expect("slot 2");
        assert_eq!(got.nid, 0xEEFF);
        assert_eq!(got.addr, UNRESOLVED_STUB_BASE + 16);
    }

    #[test]
    fn address_below_the_base_is_not_ours() {
        let stubs = vec![stub(0xAABB, 0)];
        assert!(resolve(UNRESOLVED_STUB_BASE - 8, &stubs).is_none());
        assert!(resolve(0x1234, &stubs).is_none());
    }

    #[test]
    fn address_past_the_last_stub_is_not_ours() {
        let stubs = vec![stub(0xAABB, 0), stub(0xCCDD, 1)];
        assert!(resolve(UNRESOLVED_STUB_BASE + 16, &stubs).is_none());
    }

    #[test]
    fn misaligned_address_in_range_is_a_genuine_fault_not_a_stub() {
        let stubs = vec![stub(0xAABB, 0), stub(0xCCDD, 1)];
        assert!(
            resolve(UNRESOLVED_STUB_BASE + 4, &stubs).is_none(),
            "a mid-slot address is a wild pointer, not one of our slots"
        );
    }

    #[test]
    fn empty_table_resolves_nothing() {
        assert!(resolve(UNRESOLVED_STUB_BASE, &[]).is_none());
    }
}
