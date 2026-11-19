"""Parse one annotated strain proteome; non-target header tags are `other`."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import re

from Bio import SeqIO

VALID_AA = frozenset('ACDEFGHIKLMNPQRSTVWYX')
TARGET_TYPES = frozenset({'cazyme', 'lipase', 'protease'})
TYPE_ALIASES = {
    'cazyme':'cazyme', 'cazymes':'cazyme', 'carbohydrate':'cazyme',
    'lipase':'lipase', 'lipases':'lipase',
    'protease':'protease', 'proteases':'protease', 'peptidase':'protease',
    'other':'other', 'others':'other',
    'other_protein':'other', 'other_proteins':'other', 'others_protein':'other',
    'others_proteins':'other', 'non_target':'other', 'non_targets':'other',
    'nontarget':'other', 'nontargets':'other', 'unknown':'other',
    'unclassified':'other', 'none':'other', 'unannotated':'other',
    'na':'other', '-':'other',
}
ASSEMBLY = re.compile(r'GC[AF]_\d+\.\d+', re.IGNORECASE)

@dataclass(frozen=True)
class Protein:
    protein_id: str
    sequence: str
    enzyme_type: str
    replaced: int
    header: str

    @property
    def is_target(self):
        return self.enzyme_type in TARGET_TYPES


def parse_annotated_fasta(path: str | Path, max_proteins=30000):
    path = Path(path)
    stream = path.open('r', encoding='utf-8')
    try:
        return _parse_stream(stream, path, max_proteins)
    finally:
        stream.close()

def _parse_stream(stream, path, max_proteins):
    records = []
    ids = set()
    assembly_ids = set()
    replacements = 0
    untagged = 0
    unexpected_tags = Counter()
    for rec in SeqIO.parse(stream, 'fasta'):
        if len(records) >= max_proteins:
            raise ValueError(f'Proteome exceeds the maximum of {max_proteins} proteins.')
        header = str(rec.description)
        protein_id = str(rec.id)
        if protein_id in ids:
            raise ValueError(f'Duplicate FASTA protein ID: {protein_id}')
        ids.add(protein_id)
        parts = header.split('|')
        if len(parts) >= 2:
            # Header layout: >protein_id|enzyme_type|assembly|
            # Only explicitly recognized target categories count as CAZyme,
            # lipase or protease. Any other label is a non-target protein.
            annotation = re.sub(r'[\s-]+', '_', parts[1].strip().lower())
            enzyme = TYPE_ALIASES.get(annotation, 'other')
            if annotation not in TYPE_ALIASES:
                unexpected_tags[annotation or '(empty)'] += 1
        else:
            enzyme = 'other'
            untagged += 1
        seq = re.sub(r'\s+', '', str(rec.seq)).upper().rstrip('*')
        if not seq:
            raise ValueError(f'Empty protein sequence: {protein_id}')
        if '*' in seq:
            raise ValueError(f'Internal stop codon in protein sequence: {protein_id}')
        replaced = sum(aa not in VALID_AA for aa in seq)
        if replaced:
            seq = ''.join(aa if aa in VALID_AA else 'X' for aa in seq)
        replacements += replaced
        matches = {a.upper() for a in ASSEMBLY.findall(header)}
        assembly_ids.update(matches)
        records.append(Protein(protein_id, seq, enzyme, replaced, header))
    if not records:
        raise ValueError('No protein records found in the uploaded FASTA file.')
    if len(assembly_ids) > 1:
        raise ValueError('Multiple assembly accessions detected. Upload one strain per FASTA file: ' + ', '.join(sorted(assembly_ids)[:8]))
    if not any(x.is_target for x in records):
        raise ValueError('No CAZyme, lipase, or protease headers detected. Case 4 cannot be computed. Supply annotated headers.')
    filename_assembly = ASSEMBLY.search(path.name)
    if filename_assembly and assembly_ids and filename_assembly.group().upper() not in assembly_ids:
        raise ValueError('Filename assembly accession conflicts with FASTA headers.')
    sample = sorted(assembly_ids)[0] if assembly_ids else (filename_assembly.group().upper() if filename_assembly else path.stem)
    counts = dict(Counter(x.enzyme_type for x in records))
    summary = {'analysis_id': sample, 'total_proteins':len(records), 'target_proteins':sum(x.is_target for x in records),
               'counts':counts, 'replaced_residues':replacements, 'untagged_proteins':untagged,
               'unexpected_tags_treated_as_other':dict(unexpected_tags)}
    return records, summary
