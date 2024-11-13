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

#include "lifter-private.h"

#include "instr.h"
#include "regfile.h"
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>


/**
 * \defgroup LLFlags Flags
 * \brief Computation of X86 flags
 *
 * @{
 **/

namespace rellume::x86_64 {

void Lifter::FlagCalcSAPLogic(llvm::Value* res) {
    regfile->Set(ArchReg::SF, RegFile::Transform::IsNeg, res);
    regfile->Set(ArchReg::PF, RegFile::Transform::TruncI8, res);
    SetFlagUndef({ArchReg::AF});
}

void Lifter::FlagCalcAdd(llvm::Value* res, llvm::Value* lhs,
                         llvm::Value* rhs, bool skip_carry) {
    regfile->Set(ArchReg::ZF, RegFile::Transform::IsZero, res);
    regfile->Set(ArchReg::SF, RegFile::Transform::IsNeg, res);
    regfile->Set(ArchReg::PF, RegFile::Transform::TruncI8, res);
    regfile->Set(ArchReg::AF, RegFile::Transform::X86AuxFlag, res, lhs, rhs);
    if (!skip_carry)
        regfile->Set(ArchReg::CF, RegFile::Transform::IsULT, res, lhs);

    if (cfg.enableOverflowIntrinsics) {
        llvm::Intrinsic::ID id = llvm::Intrinsic::sadd_with_overflow;
        llvm::Value* packed = irb.CreateBinaryIntrinsic(id, lhs, rhs);
        SetReg(ArchReg::OF, irb.CreateExtractValue(packed, 1));
    } else {
        regfile->Set(ArchReg::OF, RegFile::Transform::AddOverflowFlag, res, lhs, rhs);
    }
}

void Lifter::FlagCalcSub(llvm::Value* res, llvm::Value* lhs,
                         llvm::Value* rhs, bool skip_carry, bool alt_zf) {
    auto zero = llvm::Constant::getNullValue(res->getType());
    llvm::Value* sf = irb.CreateICmpSLT(res, zero);  // also used for OF

    if (alt_zf)
        SetReg(ArchReg::ZF, irb.CreateICmpEQ(lhs, rhs));
    else
        regfile->Set(ArchReg::ZF, RegFile::Transform::IsZero, res);
    SetReg(ArchReg::SF, sf);
    regfile->Set(ArchReg::PF, RegFile::Transform::TruncI8, res);
    regfile->Set(ArchReg::AF, RegFile::Transform::X86AuxFlag, res, lhs, rhs);
    if (!skip_carry)
        regfile->Set(ArchReg::CF, RegFile::Transform::IsULT, lhs, rhs);

    // Set overflow flag using arithmetic comparisons
    SetReg(ArchReg::OF, irb.CreateICmpNE(sf, irb.CreateICmpSLT(lhs, rhs)));
}

llvm::Value* Lifter::FlagCond(Condition cond) {
    llvm::Value* result = nullptr;
    switch (static_cast<Condition>(static_cast<int>(cond) & ~1)) {
    case Condition::O:  result = GetFlag(ArchReg::OF); break;
    case Condition::C:  result = GetFlag(ArchReg::CF); break;
    case Condition::Z:  result = GetFlag(ArchReg::ZF); break;
    case Condition::BE: result = irb.CreateOr(GetFlag(ArchReg::CF), GetFlag(ArchReg::ZF)); break;
    case Condition::S:  result = GetFlag(ArchReg::SF); break;
    case Condition::P:  result = GetFlag(ArchReg::PF); break;
    case Condition::L:  result = irb.CreateICmpNE(GetFlag(ArchReg::SF), GetFlag(ArchReg::OF)); break;
    case Condition::LE: result = irb.CreateOr(GetFlag(ArchReg::ZF), irb.CreateICmpNE(GetFlag(ArchReg::SF), GetFlag(ArchReg::OF))); break;
    default: assert(0);
    }

    return static_cast<int>(cond) & 1 ? irb.CreateNot(result) : result;
}

static const std::pair<ArchReg, unsigned> RFLAGS_INDICES[] = {
    {ArchReg::CF, 0}, {ArchReg::PF, 2}, {ArchReg::AF, 4}, {ArchReg::ZF, 6},
    {ArchReg::SF, 7}, {ArchReg::DF, 10}, {ArchReg::OF, 11},
};

llvm::Value* Lifter::FlagAsReg(unsigned size) {
    llvm::Value* res = irb.getInt64(0x202); // IF
    llvm::Type* ty = res->getType();
    for (auto& kv : RFLAGS_INDICES) {
        llvm::Value* ext_bit = irb.CreateZExt(GetFlag(kv.first), ty);
        res = irb.CreateOr(res, irb.CreateShl(ext_bit, kv.second));
    }
    return irb.CreateTruncOrBitCast(res, irb.getIntNTy(size));
}

void Lifter::FlagFromReg(llvm::Value* val) {
    unsigned sz = val->getType()->getIntegerBitWidth();
    for (auto& kv : RFLAGS_INDICES) {
        if (kv.second >= sz)
            break;
        llvm::Value* bit = irb.CreateLShr(val, kv.second);
        if (kv.first == ArchReg::PF) {
            bit = irb.CreateNot(irb.CreateTrunc(bit, irb.getInt1Ty()));
            SetReg(kv.first, irb.CreateZExt(bit, irb.getInt8Ty()));
        } else {
            SetReg(kv.first, irb.CreateTrunc(bit, irb.getInt1Ty()));
        }
    }
}

} // namespace::x86_64

/**
 * @}
 **/
