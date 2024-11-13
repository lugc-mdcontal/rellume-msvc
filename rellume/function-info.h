/**
 * This file is part of Rellume.
 *
 * (c) 2016-2019, Alexis Engelke <alexis.engelke@googlemail.com>
 *
 * Rellume is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License (LGPL)
 * as published by the Free Software Foundation, either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * Rellume is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Rellume.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * \file
 **/

#ifndef RELLUME_FUNCTION_INFO_H
#define RELLUME_FUNCTION_INFO_H

#include <cstdbool>
#include <cstdint>
#include <memory>
#include <vector>


namespace llvm {
class Function;
class Instruction;
class Value;
}

namespace rellume {

#ifdef RELLUME_WITH_X86_64
namespace SptrIdx::x86_64 {
    enum {
#define RELLUME_MAPPED_REG(nameu,...) nameu,
#ifdef RELLUME_MAPPED_REG
RELLUME_MAPPED_REG(RIP, 0, ArchReg::INVALID, Facet::I64)
RELLUME_MAPPED_REG(RAX, 8, ArchReg::GP(0), Facet::I64)
RELLUME_MAPPED_REG(RCX, 16, ArchReg::GP(1), Facet::I64)
RELLUME_MAPPED_REG(RDX, 24, ArchReg::GP(2), Facet::I64)
RELLUME_MAPPED_REG(RBX, 32, ArchReg::GP(3), Facet::I64)
RELLUME_MAPPED_REG(RSP, 40, ArchReg::GP(4), Facet::I64)
RELLUME_MAPPED_REG(RBP, 48, ArchReg::GP(5), Facet::I64)
RELLUME_MAPPED_REG(RSI, 56, ArchReg::GP(6), Facet::I64)
RELLUME_MAPPED_REG(RDI, 64, ArchReg::GP(7), Facet::I64)
RELLUME_MAPPED_REG(R8, 72, ArchReg::GP(8), Facet::I64)
RELLUME_MAPPED_REG(R9, 80, ArchReg::GP(9), Facet::I64)
RELLUME_MAPPED_REG(R10, 88, ArchReg::GP(10), Facet::I64)
RELLUME_MAPPED_REG(R11, 96, ArchReg::GP(11), Facet::I64)
RELLUME_MAPPED_REG(R12, 104, ArchReg::GP(12), Facet::I64)
RELLUME_MAPPED_REG(R13, 112, ArchReg::GP(13), Facet::I64)
RELLUME_MAPPED_REG(R14, 120, ArchReg::GP(14), Facet::I64)
RELLUME_MAPPED_REG(R15, 128, ArchReg::GP(15), Facet::I64)
RELLUME_MAPPED_REG(ZF, 136, ArchReg::FLAG(0), Facet::I1)
RELLUME_MAPPED_REG(SF, 137, ArchReg::FLAG(1), Facet::I1)
RELLUME_MAPPED_REG(PF, 138, ArchReg::FLAG(2), Facet::I8)
RELLUME_MAPPED_REG(CF, 139, ArchReg::FLAG(3), Facet::I1)
RELLUME_MAPPED_REG(OF, 140, ArchReg::FLAG(4), Facet::I1)
RELLUME_MAPPED_REG(AF, 141, ArchReg::FLAG(5), Facet::I1)
RELLUME_MAPPED_REG(DF, 142, ArchReg::FLAG(6), Facet::I1)
RELLUME_MAPPED_REG(FSBASE, 144, ArchReg::INVALID, Facet::I64)
RELLUME_MAPPED_REG(GSBASE, 152, ArchReg::INVALID, Facet::I64)
RELLUME_MAPPED_REG(XMM0, 160, ArchReg::VEC(0), Facet::V2I64)
RELLUME_MAPPED_REG(XMM1, 176, ArchReg::VEC(1), Facet::V2I64)
RELLUME_MAPPED_REG(XMM2, 192, ArchReg::VEC(2), Facet::V2I64)
RELLUME_MAPPED_REG(XMM3, 208, ArchReg::VEC(3), Facet::V2I64)
RELLUME_MAPPED_REG(XMM4, 224, ArchReg::VEC(4), Facet::V2I64)
RELLUME_MAPPED_REG(XMM5, 240, ArchReg::VEC(5), Facet::V2I64)
RELLUME_MAPPED_REG(XMM6, 256, ArchReg::VEC(6), Facet::V2I64)
RELLUME_MAPPED_REG(XMM7, 272, ArchReg::VEC(7), Facet::V2I64)
RELLUME_MAPPED_REG(XMM8, 288, ArchReg::VEC(8), Facet::V2I64)
RELLUME_MAPPED_REG(XMM9, 304, ArchReg::VEC(9), Facet::V2I64)
RELLUME_MAPPED_REG(XMM10, 320, ArchReg::VEC(10), Facet::V2I64)
RELLUME_MAPPED_REG(XMM11, 336, ArchReg::VEC(11), Facet::V2I64)
RELLUME_MAPPED_REG(XMM12, 352, ArchReg::VEC(12), Facet::V2I64)
RELLUME_MAPPED_REG(XMM13, 368, ArchReg::VEC(13), Facet::V2I64)
RELLUME_MAPPED_REG(XMM14, 384, ArchReg::VEC(14), Facet::V2I64)
RELLUME_MAPPED_REG(XMM15, 400, ArchReg::VEC(15), Facet::V2I64)
#endif
#ifdef RELLUME_NAMED_REG
RELLUME_NAMED_REG(rip, RIP, 8, 0)
RELLUME_NAMED_REG(rax, RAX, 8, 8)
RELLUME_NAMED_REG(rcx, RCX, 8, 16)
RELLUME_NAMED_REG(rdx, RDX, 8, 24)
RELLUME_NAMED_REG(rbx, RBX, 8, 32)
RELLUME_NAMED_REG(rsp, RSP, 8, 40)
RELLUME_NAMED_REG(rbp, RBP, 8, 48)
RELLUME_NAMED_REG(rsi, RSI, 8, 56)
RELLUME_NAMED_REG(rdi, RDI, 8, 64)
RELLUME_NAMED_REG(r8, R8, 8, 72)
RELLUME_NAMED_REG(r9, R9, 8, 80)
RELLUME_NAMED_REG(r10, R10, 8, 88)
RELLUME_NAMED_REG(r11, R11, 8, 96)
RELLUME_NAMED_REG(r12, R12, 8, 104)
RELLUME_NAMED_REG(r13, R13, 8, 112)
RELLUME_NAMED_REG(r14, R14, 8, 120)
RELLUME_NAMED_REG(r15, R15, 8, 128)
RELLUME_NAMED_REG(zf, ZF, 1, 136)
RELLUME_NAMED_REG(sf, SF, 1, 137)
RELLUME_NAMED_REG(pf, PF, 1, 138)
RELLUME_NAMED_REG(cf, CF, 1, 139)
RELLUME_NAMED_REG(of, OF, 1, 140)
RELLUME_NAMED_REG(af, AF, 1, 141)
RELLUME_NAMED_REG(df, DF, 1, 142)
RELLUME_NAMED_REG(fsbase, FSBASE, 8, 144)
RELLUME_NAMED_REG(gsbase, GSBASE, 8, 152)
RELLUME_NAMED_REG(xmm0, XMM0, 16, 160)
RELLUME_NAMED_REG(xmm1, XMM1, 16, 176)
RELLUME_NAMED_REG(xmm2, XMM2, 16, 192)
RELLUME_NAMED_REG(xmm3, XMM3, 16, 208)
RELLUME_NAMED_REG(xmm4, XMM4, 16, 224)
RELLUME_NAMED_REG(xmm5, XMM5, 16, 240)
RELLUME_NAMED_REG(xmm6, XMM6, 16, 256)
RELLUME_NAMED_REG(xmm7, XMM7, 16, 272)
RELLUME_NAMED_REG(xmm8, XMM8, 16, 288)
RELLUME_NAMED_REG(xmm9, XMM9, 16, 304)
RELLUME_NAMED_REG(xmm10, XMM10, 16, 320)
RELLUME_NAMED_REG(xmm11, XMM11, 16, 336)
RELLUME_NAMED_REG(xmm12, XMM12, 16, 352)
RELLUME_NAMED_REG(xmm13, XMM13, 16, 368)
RELLUME_NAMED_REG(xmm14, XMM14, 16, 384)
RELLUME_NAMED_REG(xmm15, XMM15, 16, 400)
#endif
#undef RELLUME_MAPPED_REG
    };
}
#endif // RELLUME_WITH_X86_64
#ifdef RELLUME_WITH_RV64
namespace SptrIdx::rv64 {
    enum {
#define RELLUME_MAPPED_REG(nameu,...) nameu,
#include <rellume/cpustruct-rv64-private.inc>
#undef RELLUME_MAPPED_REG
    };
}
#endif // RELLUME_WITH_RV64
#ifdef RELLUME_WITH_AARCH64
namespace SptrIdx::aarch64 {
    enum {
#define RELLUME_MAPPED_REG(nameu,...) nameu,
#include <rellume/cpustruct-aarch64-private.inc>
#undef RELLUME_MAPPED_REG
    };
}
#endif // RELLUME_WITH_AARCH64

class ArchBasicBlock;
class RegFile;

/// CallConvPack records which registers were changed in a basic block,
/// and pre-computed LLVM store instructions for them.
///
/// After the function is lifted, its packs are optimised to avoid
/// passing unnecessary store instructions to the LLVM optimiser, which
/// would struggle with the many superfluous stores.
struct CallConvPack {
    std::unique_ptr<RegFile> regfile;
    llvm::Instruction* packBefore;
    ArchBasicBlock* bb;
};

/// FunctionInfo holds the LLVM objects of the lifted function and its
/// environment: sptr is the single argument of the function and stands
/// for "CPU struct pointer". A CPU struct stores a register set. See
/// data/rellume/*cpu.json for the architecture-dependent definitions.
///
/// The CPU struct concept allows passing arguments and returning values
/// without requiring knowledge of the calling convention or the function
/// signature.
struct FunctionInfo {
    /// The function itself
    llvm::Function* fn;
    /// The sptr argument, and its elements
    llvm::Value* sptr_raw;
    std::vector<llvm::Value*> sptr;

    uint64_t pc_base_addr;
    llvm::Value* pc_base_value;

    std::vector<CallConvPack> call_conv_packs;
};


} // namespace

#endif
