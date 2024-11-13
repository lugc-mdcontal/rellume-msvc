/**
 * This file is part of Rellume.
 *
 * (c) 2019, Alexis Engelke <alexis.engelke@googlemail.com>
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

#include "callconv.h"

#include "basicblock.h"
#include "function-info.h"
#include "regfile.h"
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <cassert>


namespace rellume {

CallConv CallConv::FromFunction(llvm::Function* fn, Arch arch) {
    auto fn_cconv = fn->getCallingConv();
    auto fn_ty = fn->getFunctionType();
    CallConv hunch = INVALID;
    switch (arch) {
#ifdef RELLUME_WITH_X86_64
    case Arch::X86_64: hunch = X86_64_SPTR; break;
#endif // RELLUME_WITH_X86_64
#ifdef RELLUME_WITH_RV64
    case Arch::RV64: hunch = RV64_SPTR; break;
#endif // RELLUME_WITH_RV64
#ifdef RELLUME_WITH_AARCH64
    case Arch::AArch64: hunch = AArch64_SPTR; break;
#endif // RELLUME_WITH_AARCH64
    default:
        return INVALID;
    }

    // Verify hunch.
    if (hunch.FnCallConv() != fn_cconv)
        return INVALID;
    unsigned sptr_idx = hunch.CpuStructParamIdx();
    if (sptr_idx >= fn->arg_size())
        return INVALID;
    llvm::Type* sptr_ty = fn->arg_begin()[sptr_idx].getType();
    if (!sptr_ty->isPointerTy())
        return INVALID;
    unsigned sptr_addrspace = sptr_ty->getPointerAddressSpace();
    if (fn_ty != hunch.FnType(fn->getContext(), sptr_addrspace))
        return INVALID;
    return hunch;
}

llvm::FunctionType* CallConv::FnType(llvm::LLVMContext& ctx,
                                     unsigned sptr_addrspace) const {
    llvm::Type* void_ty = llvm::Type::getVoidTy(ctx);
    llvm::Type* ptrTy = llvm::PointerType::get(ctx, sptr_addrspace);

    switch (*this) {
    default:
        return nullptr;
    case CallConv::X86_64_SPTR:
    case CallConv::RV64_SPTR:
    case CallConv::AArch64_SPTR:
        return llvm::FunctionType::get(void_ty, {ptrTy}, false);
    }
}

llvm::CallingConv::ID CallConv::FnCallConv() const {
    switch (*this) {
    default: return llvm::CallingConv::C;
    case CallConv::X86_64_SPTR: return llvm::CallingConv::C;
    case CallConv::RV64_SPTR: return llvm::CallingConv::C;
    case CallConv::AArch64_SPTR: return llvm::CallingConv::C;
    }
}

unsigned CallConv::CpuStructParamIdx() const {
    switch (*this) {
    default: return 0;
    case CallConv::X86_64_SPTR:  return 0;
    case CallConv::RV64_SPTR:    return 0;
    case CallConv::AArch64_SPTR: return 0;
    }
}

Arch CallConv::ToArch() const {
    switch (*this) {
    default: return Arch::INVALID;
    case CallConv::X86_64_SPTR:  return Arch::X86_64;
    }
}

using CPUStructEntry = std::tuple<unsigned, unsigned, ArchReg, Facet>;

// Note: replace with C++20 std::span.
template<typename T>
class span {
    T* ptr;
    std::size_t len;
public:
    constexpr span() : ptr(nullptr), len(0) {}
    template<std::size_t N>
    constexpr span(T (&arr)[N]) : ptr(arr), len(N) {}
    constexpr std::size_t size() const { return len; }
    constexpr T* begin() const { return &ptr[0]; }
    constexpr T* end() const { return &ptr[len]; }
};

static span<const CPUStructEntry> CPUStructEntries(CallConv cconv) {
#ifdef RELLUME_WITH_X86_64
    static const CPUStructEntry cpu_struct_entries_x86_64[] = {
#define RELLUME_MAPPED_REG(nameu,off,reg,facet) \
            std::make_tuple(SptrIdx::x86_64::nameu, off, reg, facet),
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
#endif // RELLUME_WITH_X86_64

#ifdef RELLUME_WITH_RV64
    static const CPUStructEntry cpu_struct_entries_rv64[] = {
#define RELLUME_MAPPED_REG(nameu,off,reg,facet) \
            std::make_tuple(SptrIdx::rv64::nameu, off, reg, facet),
#include <rellume/cpustruct-rv64-private.inc>
#undef RELLUME_MAPPED_REG
    };
#endif // RELLUME_WITH_RV64

#ifdef RELLUME_WITH_AARCH64
    static const CPUStructEntry cpu_struct_entries_aarch64[] = {
#define RELLUME_MAPPED_REG(nameu,off,reg,facet) \
            std::make_tuple(SptrIdx::aarch64::nameu, off, reg, facet),
#include <rellume/cpustruct-aarch64-private.inc>
#undef RELLUME_MAPPED_REG
    };
#endif // RELLUME_WITH_AARCH64

    switch (cconv) {
    default:
        return span<const CPUStructEntry>();
#ifdef RELLUME_WITH_X86_64
    case CallConv::X86_64_SPTR:
        return cpu_struct_entries_x86_64;
#endif // RELLUME_WITH_X86_64
#ifdef RELLUME_WITH_RV64
    case CallConv::RV64_SPTR:
        return cpu_struct_entries_rv64;
#endif // RELLUME_WITH_RV64
#ifdef RELLUME_WITH_AARCH64
    case CallConv::AArch64_SPTR:
        return cpu_struct_entries_aarch64;
#endif // RELLUME_WITH_AARCH64
    }
}


void CallConv::InitSptrs(ArchBasicBlock* bb, FunctionInfo& fi) {
    llvm::IRBuilder<> irb(bb->BeginBlock());
    llvm::Type* i8 = irb.getInt8Ty();

    const auto& cpu_struct_entries = CPUStructEntries(*this);
    fi.sptr.resize(cpu_struct_entries.size());
    for (const auto& [sptr_idx, off, reg, facet] : cpu_struct_entries)
        fi.sptr[sptr_idx] = irb.CreateConstGEP1_64(i8, fi.sptr_raw, off);
}

static void Pack(ArchBasicBlock* bb, FunctionInfo& fi, llvm::Instruction* before) {
    CallConvPack& pack_info = fi.call_conv_packs.emplace_back();
    pack_info.regfile = bb->TakeRegFile();
    pack_info.packBefore = before;
    pack_info.bb = bb;
}

template<typename F>
static void Unpack(CallConv cconv, ArchBasicBlock* bb, llvm::BasicBlock* llvmBlock, FunctionInfo& fi, F get_from_reg) {
    bb->InitEmpty(cconv.ToArch(), llvmBlock);
    // New regfile with everything cleared
    RegFile& regfile = *bb->GetRegFile();
    llvm::IRBuilder<> irb(regfile.GetInsertBlock());

    regfile.SetPC(irb.CreateLoad(irb.getInt64Ty(), fi.sptr_raw));
    for (const auto& [sptr_idx, off, reg, facet] : CPUStructEntries(cconv)) {
        if (reg.Kind() == ArchReg::RegKind::INVALID)
            continue;
        if (llvm::Value* reg_val = get_from_reg(reg)) {
            regfile.Set(reg, reg_val);
            continue;
        }

        llvm::Type* reg_ty = facet.Type(irb.getContext());
        auto typeVal = llvm::Constant::getNullValue(reg_ty);
        // Mark register as clean if it was loaded from the sptr.
        regfile.Set(reg, RegFile::Transform::Load, fi.sptr[sptr_idx], typeVal);
        regfile.DirtyRegs()[RegisterSetBitIdx(reg)] = false;
    }
}

llvm::ReturnInst* CallConv::Return(ArchBasicBlock* bb, FunctionInfo& fi) const {
    llvm::IRBuilder<> irb(bb->GetRegFile()->GetInsertBlock());
    llvm::ReturnInst* ret = irb.CreateRetVoid();
    Pack(bb, fi, ret);
    return ret;
}

void CallConv::UnpackParams(ArchBasicBlock* bb, FunctionInfo& fi) const {
    Unpack(*this, bb, bb->BeginBlock(), fi, [&fi] (ArchReg reg) {
        return nullptr;
    });
}

llvm::CallInst* CallConv::Call(llvm::Function* fn, ArchBasicBlock* bb,
                               FunctionInfo& fi, bool tail_call) {
    llvm::SmallVector<llvm::Value*, 16> call_args;
    call_args.resize(fn->arg_size());
    call_args[CpuStructParamIdx()] = fi.sptr_raw;

    llvm::IRBuilder<> irb(bb->EndBlock());

    llvm::CallInst* call = irb.CreateCall(fn->getFunctionType(), fn, call_args);
    call->setCallingConv(fn->getCallingConv());
    call->setAttributes(fn->getAttributes());

    Pack(bb, fi, call);

    if (tail_call) {
        call->setTailCallKind(llvm::CallInst::TCK_MustTail);
        if (call->getType()->isVoidTy())
            irb.CreateRetVoid();
        else
            irb.CreateRet(call);
        return call;
    }

    Unpack(*this, bb, irb.GetInsertBlock(), fi, [] (ArchReg reg) {
        return nullptr;
    });

    return call;
}

void CallConv::OptimizePacks(FunctionInfo& fi, ArchBasicBlock* entry) {
    // Map of basic block to dirty register at (beginning, end) of the block.
    llvm::DenseMap<ArchBasicBlock*, std::pair<RegisterSet, RegisterSet>> bb_map;

    llvm::SmallPtrSet<ArchBasicBlock*, 16> queue;
    llvm::SmallVector<ArchBasicBlock*, 16> queue_vec;
    queue.insert(entry);
    queue_vec.push_back(entry);

    while (!queue.empty()) {
        llvm::SmallVector<ArchBasicBlock*, 16> new_queue_vec;
        for (ArchBasicBlock* bb : queue_vec) {
            queue.erase(bb);
            RegisterSet pre;
            for (ArchBasicBlock* pred : bb->Predecessors())
                pre |= bb_map.lookup(pred).second;

            RegisterSet post;
            if (RegFile* rf = bb->GetRegFile()){
                if (rf->StartsClean())
                    post = rf->DirtyRegs();
                else
                    post = pre | rf->DirtyRegs();
            }
            auto new_regsets = std::make_pair(pre, post);

            auto [it, inserted] = bb_map.try_emplace(bb, new_regsets);
            // If it is the first time we look at bb, or the set of dirty
            // registers changed, look at successors (again).
            if (inserted || it->second.second != post) {
                for (ArchBasicBlock* succ : bb->Successors()) {
                    // If user not in the set, then add it to the vector.
                    if (queue.insert(succ).second)
                        new_queue_vec.push_back(succ);
                }
            }

            // Ensure that we store the new value.
            if (!inserted)
                it->second = new_regsets;
        }
        queue_vec = std::move(new_queue_vec);
    }

    for (const auto& pack : fi.call_conv_packs) {
        RegFile& regfile = *pack.regfile;
        regfile.SetInsertPoint(pack.packBefore->getIterator());

        RegisterSet regset = regfile.DirtyRegs();
        if (!regfile.StartsClean())
            regset |= bb_map.lookup(pack.bb).first;
        llvm::IRBuilder<> irb(pack.packBefore);
        irb.CreateStore(regfile.GetPCValue(fi.pc_base_value, fi.pc_base_addr), fi.sptr_raw);
        for (const auto& [sptr_idx, off, reg, facet] : CPUStructEntries(*this)) {
            if (reg.Kind() == ArchReg::RegKind::INVALID)
                continue;
            unsigned regidx = RegisterSetBitIdx(reg);
            if (!regset[regidx])
                continue;
            // Find best position for store. Hoist stores up to predecessors
            // where possible to avoid executing stores on code paths that never
            // write the register, but don't hoist them inside loops or similar.
            ArchBasicBlock* bb = pack.bb;
            RegFile* rf = &regfile;
            while (!rf->StartsClean() && !rf->DirtyRegs()[regidx]) {
                // Try to find single predecessor where register is written.
                ArchBasicBlock* dirtyPred = nullptr;
                for (ArchBasicBlock* pred : bb->Predecessors()) {
                    // Ignore predecessors where the register is never written.
                    if (!bb_map.lookup(pred).second[regidx])
                        continue;
                    if (dirtyPred) {
                        dirtyPred = nullptr;
                        break;
                    }
                    dirtyPred = pred;
                }
                // If there is no single dirty predecessor or if that has
                // multiple successors (possibly a loop), abort.
                if (!dirtyPred || dirtyPred->Successors().size() != 1)
                    break;

                bb = dirtyPred;
                rf = bb->GetRegFile();
            }

            llvm::Value* reg_val = rf->GetReg(reg, facet);
            if (llvm::isa<llvm::UndefValue>(reg_val))
                continue; // Just remove stores of undef.
            if (rf != &regfile) {
                auto terminator = rf->GetInsertBlock()->getTerminator();
                new llvm::StoreInst(reg_val, fi.sptr[sptr_idx], terminator);
            } else {
                irb.CreateStore(reg_val, fi.sptr[sptr_idx]);
            }
        }
    }
}

} // namespace
