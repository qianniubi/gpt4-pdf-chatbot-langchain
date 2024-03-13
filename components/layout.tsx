interface LayoutProps {
  children?: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="mx-auto flex flex-col space-y-4">
      <header className="container sticky top-0 z-40 bg-white" style={{width: '100vw'}}>
        <div className="h-16 border-b border-b-slate-200 py-4">
          <nav className="ml-0 pl-0">
            <p className="text-center font-bold text-2xl">
              AI Counselor
            </p>
          </nav>
        </div>
      </header>
      <div>
        <main className="flex w-full flex-1 flex-col overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}
