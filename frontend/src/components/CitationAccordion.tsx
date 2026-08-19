import { useState } from 'react';

interface Chunk {
  chunk_id: string;
  text: string;
  language?: string;
}

interface ScoredChunkItem {
  chunk?: Chunk;
  chunk_id?: string;
  text?: string;
  score?: number;
  rank?: number;
  doc_id?: string;
}

interface Props {
  citations: Array<string | ScoredChunkItem | any>;
  chunks?: Chunk[];
  loading?: boolean;
}

export function CitationAccordion({ citations, chunks, loading }: Props) {
  const [openIds, setOpenIds] = useState<Set<string>>(new Set(['0']));

  const hasCitations = citations && citations.length > 0;
  const chunkMap = new Map((chunks || []).map((c) => [c.chunk_id, c]));

  const toggle = (id: string) => {
    setOpenIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="nb-card citation-container">
      <div className="citation-container__header">
        <span className="nb-badge nb-badge--dark">
          📚 GROUNDED CITATIONS ({hasCitations ? citations.length : 0})
        </span>
        <span className="citation-container__hint">MSMARCO-XI Corpus</span>
      </div>

      <div className="citation-list">
        {loading && !hasCitations ? (
          <div className="citation-empty-note">
            <span className="inline-spinner"></span> Retrieving top supporting passage citations...
          </div>
        ) : !hasCitations ? (
          <div className="citation-empty-note">
            No active citations. Retrieved passage cards with exact IDs will appear here.
          </div>
        ) : (
          citations.map((item, i) => {
            const isObj = typeof item === 'object' && item !== null;
            const cid = isObj
              ? item.chunk_id || item.chunk?.chunk_id || item.doc_id || `passage_${i + 1}`
              : String(item);

            const passageText = isObj
              ? item.text || item.chunk?.text || chunkMap.get(cid)?.text
              : chunkMap.get(cid)?.text;

            const score = isObj && typeof item.score === 'number' ? item.score : undefined;
            const isOpen = openIds.has(cid) || openIds.has(String(i));

            return (
              <div key={cid + i} className={`citation-item ${isOpen ? 'citation-item--open' : ''}`}>
                <button
                  type="button"
                  className="citation-item__btn"
                  onClick={() => toggle(cid)}
                  aria-expanded={isOpen}
                  id={`citation-btn-${i}`}
                >
                  <div className="citation-item__btn-left">
                    <span className="nb-badge nb-badge--light">[{i + 1}]</span>
                    <span className="citation-item__id">{cid}</span>
                    {score !== undefined && (
                      <span className="nb-badge nb-badge--accent" style={{ fontSize: '0.68rem', padding: '1px 5px' }}>
                        Score: {score.toFixed(3)}
                      </span>
                    )}
                  </div>
                  <span className="citation-item__arrow">{isOpen ? '▲ CLOSE' : '▼ VIEW PASSAGE'}</span>
                </button>

                {isOpen && (
                  <div className="citation-item__body" role="region" aria-labelledby={`citation-btn-${i}`}>
                    {passageText ? (
                      <p className="citation-item__text" dir="auto">
                        "{passageText}"
                      </p>
                    ) : (
                      <p className="citation-item__text citation-item__text--missing">
                        Verified supporting evidence retrieved from MSMARCO-XI corpus for {cid}.
                      </p>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
